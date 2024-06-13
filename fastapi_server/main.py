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
LOG_FILE_PATH = "./app/logs/error.log"
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

try:
    STANDARD_SAMPLING_RATE = int(os.environ["STANDARD_SAMPLING_RATE"])
    STANDARD_BATCH_SIZE = int(os.environ["STANDARD_BATCH_SIZE"])
    INFERENCE_SERVER_HOST = os.environ["INFERENCE_SERVER_HOST"]
    DEFAULT_API_KEY_VALUE = os.environ["DEFAULT_API_KEY_VALUE"]
    # LOGGER_DB_PATH = os.environ["LOGGER_DB_PATH"]
    # Logging setup
    ENABLE_LOGGING = os.environ.get("ENABLE_LOGGING", "false").lower() == "true"
    if ENABLE_LOGGING:
        LOGGER_LOCAL_PATH = "./app/logs"  # Change this to your local logs directory
except KeyError as e:
    logger.error(f"Environment variable {str(e)} not found")
    raise

## Initialize Triton client for a worker ##
try:
    from inference_client import InferenceClient
    inference_client = InferenceClient(INFERENCE_SERVER_HOST)
except ImportError as e:
    logger.error(f"Error importing InferenceClient: {e}")
    raise

## Create FastAPI app ##

def AuthProvider(
    request: Request,
    credentials_key: str = Depends(APIKeyHeader(name="Authorization")),
):
    try:
        validate_status = credentials_key and credentials_key == DEFAULT_API_KEY_VALUE
        if not validate_status:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"message": "Not authenticated"},
            )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
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

## Helper functions for logging ##
def save_audio_log(audio_log_path, audio_log_entry):
    try:
        with open(audio_log_path, "a") as audio_file:
            try:
                audio_file.write(json.dumps(audio_log_entry) + "\n")
            except Exception as e:
                logger.error(f"Error writing to audio log file: {e}")
                raise
    except Exception as e:
        logger.error(f"Error opening audio log file: {e}")

def save_metadata_log(metadata_log_path, result_json):
    try:
        with open(metadata_log_path, "a") as metadata_file:
            try:
                json.dump(result_json, metadata_file, ensure_ascii=False, indent=4)
                metadata_file.write(",\n")
            except Exception as e:
                logger.error(f"Error writing to metadata log file: {e}")
                raise
    except Exception as e:
        logger.error(f"Error opening metadata log file: {e}")

## API Endpoints ##
@api.post("/api", response_model=InferenceResponse)
async def inference(request: InferenceRequest, response: Response):
    try:
        language = request.config.language.sourceLanguage
        enable_logging = ENABLE_LOGGING  # and request.controlConfig.dataTracking
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
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            metadata = {
                "timestamp": current_timestamp,
                "input_id": f"{current_timestamp}/{shortuuid.uuid()}",
                "language": language,
            }

            if enable_logging:
                # Create a date-wise directory for logs
                logs_base_dir = os.path.join(LOGGER_LOCAL_PATH, date_str)
                audio_log_path = os.path.join(logs_base_dir, "audio.log")

                # Create directory if it doesn't exist
                os.makedirs(logs_base_dir, exist_ok=True)

                # Write audio log entry
                audio_log_entry = {
                    "Id": metadata["input_id"],
                    "base64": base64.b64encode(file_bytes).decode('utf-8'),
                    "language": language
                }
                try:
                    save_audio_log(audio_log_path, audio_log_entry)
                except Exception as e:
                    logger.error(f"Error saving audio log: {e}")

            raw_audio = get_raw_audio_from_file_bytes(file_bytes, standard_sampling_rate=STANDARD_SAMPLING_RATE)

            # Check for empty audio file
            if len(raw_audio) == 0:
                logger.error(f"Empty audio file detected: {metadata['input_id']}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"message": f"Empty audio file detected for input_index: {input_index}"},
                )

            # For now, audio is small in size from NPCI, hence no VAD is required. So proceed directly without splitting
            raw_audio_list.append(raw_audio)
            metadata_list.append(metadata)
        
        final_results = []
        batches = batchify(raw_audio_list, batch_size=STANDARD_BATCH_SIZE)
        for i in range(len(batches)):
            try:
                batch_result = inference_client.run_batch_inference(batch=batches[i], lang_code=language, batch_size=STANDARD_BATCH_SIZE)
                
                for item_index, result_json in enumerate(batch_result):
                    input_index = i * STANDARD_BATCH_SIZE + item_index

                    # Check for empty transcript
                    if not result_json["transcript"]:
                        logger.error(f"Empty transcript detected: {metadata_list[input_index]['input_id']}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": f"Empty transcript detected for input_index: {input_index}"},
                        )

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
                        metadata_list[input_index]["result"] = result.model_dump(mode="json")
                        result_json = metadata_list[input_index]["result"]
                        result_json["language"] = language

                        # Ensure the source text is decoded properly from Unicode
                        if "source" in result_json:
                            utf8_content = result_json["source"].encode('utf-8')
                            result_json["source"] = utf8_content.decode('utf-8')

                        metadata_log_path = os.path.join(LOGGER_LOCAL_PATH, date_str, "response.log")
                        try:
                            save_metadata_log(metadata_log_path, result_json)
                        except Exception as e:
                            logger.error(f"Error saving metadata log: {e}")

            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"Error processing batch: {e}")

        return InferenceResponse(
            output=final_results,
            status=ResponseStatus(success=True),
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in inference endpoint: {e}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return InferenceResponse(
            output=[],
            status=ResponseStatus(
                success=False,
                message="Internal server error"
            ),
        )
