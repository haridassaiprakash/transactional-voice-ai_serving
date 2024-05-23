import os
import io
import json
import base64
import datetime
import shortuuid
import logging
from urllib.request import urlopen

from fastapi import FastAPI, Depends, Header, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader, HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from schema import *
from utils import batchify, get_raw_audio_from_file_bytes

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a file handler
LOG_FILE_PATH = os.environ["LOG_FILE_PATH"]
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.ERROR)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# ## Read environment variables only during development purpose ##
# from dotenv import load_dotenv
# load_dotenv()

STANDARD_SAMPLING_RATE = int(os.environ["STANDARD_SAMPLING_RATE"])
STANDARD_BATCH_SIZE = int(os.environ["STANDARD_BATCH_SIZE"])
INFERENCE_SERVER_HOST = os.environ["INFERENCE_SERVER_HOST"]
DEFAULT_API_KEY_VALUE = os.environ["DEFAULT_API_KEY_VALUE"]
# LOGGER_DB_PATH = os.environ["LOGGER_DB_PATH"]
# Logging setup
ENABLE_LOGGING = os.environ.get("ENABLE_LOGGING", "false").lower() == "true"
if ENABLE_LOGGING:
    LOGGER_LOCAL_PATH = "./app/metadata_log"  # Change this to your local logs directory

## Initialize Triton client for a worker ##
from inference_client import InferenceClient
inference_client = InferenceClient(INFERENCE_SERVER_HOST)

## Create FastAPI app ##

def AuthProvider(
    request: Request,
    credentials_key: str = Depends(APIKeyHeader(name="Authorization")),
):
    validate_status = credentials_key and credentials_key == DEFAULT_API_KEY_VALUE
    if not validate_status:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Not authenticated"},
        )


api = FastAPI(
    title="NPCI ASR Inference API",
    description="Backend API for communicating with ASR models",
    dependencies=[
        # Depends(AuthProvider),
    ],
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## API Endpoints ##
@api.post("/api", response_model=InferenceResponse)
async def inference(request: InferenceRequest, response: Response):
    language = request.config.language.sourceLanguage
    enable_logging = ENABLE_LOGGING # and request.controlConfig.dataTracking
    raw_audio_list, metadata_list = [], []

    for input_index, input_item in enumerate(request.audio):
        
        if input_item.audioContent:
            file_bytes = base64.b64decode(input_item.audioContent)
        elif input_item.audioUri:
            try:
                file_bytes = urlopen(input_item.audioUri).read()
            except Exception as e:
                logger.error(f"Error fetching audio from URI: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"message": f"Error fetching audio from URI: {e}"}
                )
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return InferenceResponse(
                status=ResponseStatus(
                    success=False,
                    message=f"Neither `audioContent` nor `audioUri` found in `audio` input_index: {input_index}",
                ),
            )
        
        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
        metadata = {
            "timestamp": current_timestamp,
            "input_id": f"{current_timestamp}/{shortuuid.uuid()}",
            "language": language,
        }

        if enable_logging:
            try:
                # Create unique directories for each audio and metadata log
                logs_base_dir = os.path.join(LOGGER_LOCAL_PATH, metadata['input_id'])
                audio_log_path = os.path.join(logs_base_dir, f"audio.{request.config.audioFormat}")
                metadata_log_path = os.path.join(logs_base_dir, "metadata.json")

                # Create directories if they don't exist
                os.makedirs(logs_base_dir, exist_ok=True)

                # Write audio log
                with open(audio_log_path, "wb") as audio_file:
                    audio_file.write(file_bytes)

                # Write metadata log
                with open(metadata_log_path, "w") as metadata_file:
                    json.dump(metadata, metadata_file, indent=4)
            except Exception as e:
                logger.error(f"Error saving logs locally: {e}")

        raw_audio = get_raw_audio_from_file_bytes(file_bytes, standard_sampling_rate=STANDARD_SAMPLING_RATE)

        # For now, audio is small in size from NPCI, hence no VAD is required. So proceed directly without splitting
        raw_audio_list.append(raw_audio)
        metadata_list.append(metadata)
    
    final_results = []
    batches = batchify(raw_audio_list, batch_size=STANDARD_BATCH_SIZE)
    for i in range(len(batches)):
        try:
            batch_result = inference_client.run_batch_inference(batch=batches[i], lang_code=language, batch_size=STANDARD_BATCH_SIZE)
            
            for item_index, result_json in enumerate(batch_result):
                input_index = i*STANDARD_BATCH_SIZE + item_index

                # Convert intermediate format to final format
                if "tag_entities" in request.config.postProcessors:
                    result = InferenceResult(
                        id=metadata_list[input_index]["input_id"],
                        source=result_json["transcript"],
                        entities=[
                            Entity(
                                entity=entity["entity"],
                                word=entity["word"],
                                start=entity["start"],
                                end=entity["end"],
                                value=entity["value"]
                            ) for entity in result_json["entities"]
                        ],
                        intent=result_json["intent"]
                    )
                else:
                    result = InferenceResult(
                        id=metadata_list[input_index]["input_id"],
                        source=result_json["transcript"])
                final_results.append(result)

                if enable_logging:
                    try:
                        metadata_list[input_index]["result"] = result.model_dump(mode="json")
                        result_json = metadata_list[input_index]["result"] 
                        result_json["language"] = language
                        metadata_log_path = os.path.join(logs_base_dir, "metadata.json")
                        with open(metadata_log_path, "w") as metadata_file:
                            json.dump(result_json, metadata_file, indent=4)
                    except Exception as e:
                        logger.error(f"Error saving metadata locally: {e}")
        except Exception as e:
            logger.error(f"Error processing batch: {e}")

    
    return InferenceResponse(
        output=final_results,
        status=ResponseStatus(success=True),
    )

