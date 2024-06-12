import json
import tritonclient.grpc as grpc_client
import numpy as np
import logging

from utils import pad_batch

logger = logging.getLogger(__name__)

class InferenceClient:
    def __init__(self, server_host: str) -> None:
        try:
            self._client = grpc_client.InferenceServerClient(url=server_host, verbose=False)
        except Exception as e:
            logger.error(f"Error initializing InferenceServerClient: {e}")
            raise

    def run_batch_inference(self, batch: list, lang_code: str, batch_size: int = 1) -> list:
        try:
            if batch_size == 1:
                audio_signal = np.array(batch)
                audio_len = np.asarray([[len(audio_signal[0])]], dtype=np.int32)
            else:
                audio_signal, audio_len = pad_batch(batch)
        except Exception as e:
            logger.error(f"Error preparing batch for inference: {e}")
            raise

        try:
            input0 = grpc_client.InferInput("AUDIO_SIGNAL", audio_signal.shape, "FP32")
            input0.set_data_from_numpy(audio_signal)
            input1 = grpc_client.InferInput("NUM_SAMPLES", audio_len.shape, "INT32")
            input1.set_data_from_numpy(audio_len.astype('int32'))
            output0 = grpc_client.InferRequestedOutput('TRANSCRIPTS_ASR')
            output1 = grpc_client.InferRequestedOutput('TRANSCRIPTS_ITN')
            output2 = grpc_client.InferRequestedOutput('LABELS_INTENT')
            output3 = grpc_client.InferRequestedOutput('JSON_ENTITY')
            outputs = [output0, output1, output2, output3]
            inputs = [input0, input1]
        except Exception as e:
            logger.error(f"Error creating inputs and outputs for inference: {e}")
            raise

        try:
            if lang_code == "en":
                response = self._client.infer("pipeline_pyctc_ensemble_EN", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "hi":
                response = self._client.infer("pipeline_pyctc_ensemble_HI", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "mr":
                response = self._client.infer("pipeline_pyctc_ensemble_MR", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "te":
                response = self._client.infer("pipeline_pyctc_ensemble_TE", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "or":
                response = self._client.infer("pipeline_pyctc_ensemble_OR", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "gu":
                response = self._client.infer("pipeline_pyctc_ensemble_GU", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "kn":
                response = self._client.infer("pipeline_pyctc_ensemble_KN", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "ml":
                response = self._client.infer("pipeline_pyctc_ensemble_ML", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "pa":
                response = self._client.infer("pipeline_pyctc_ensemble_PA", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "ta":
                response = self._client.infer("pipeline_pyctc_ensemble_TA", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            elif lang_code == "bn":
                response = self._client.infer("pipeline_pyctc_ensemble_BN", model_version='1', inputs=inputs, request_id=str(1), outputs=outputs)
            else:
                return [{
                    "transcript": "",
                    "transcript_itn": "",
                    "intent": "unsupported_lang",
                    "entities": [],
                }]
            
            _ = response.get_response()
        except Exception as e:
            logger.error(f"Error during inference for language {lang_code}: {e}")
            return [{
                "transcript": "",
                "transcript_itn": "",
                "intent": "error_during_inference",
                "entities": [],
            }]
        
        try:
            batch_result_asr = response.as_numpy("TRANSCRIPTS_ASR")
            batch_result_asr_itn = response.as_numpy("TRANSCRIPTS_ITN")
            batch_result_intent = response.as_numpy('LABELS_INTENT')
            batch_result_entity = response.as_numpy('JSON_ENTITY')
        except Exception as e:
            logger.error(f"Error retrieving results from inference response: {e}")
            return [{
                "transcript": "",
                "transcript_itn": "",
                "intent": "error_retrieving_results",
                "entities": [],
            }]
        
        results_json = []
        try:
            for i, sample_result in enumerate(batch_result_asr):
                result_transcript = sample_result[0].decode("utf-8")
                result_transcript_itn = batch_result_asr_itn[i][0].decode("utf-8")
                result_intent = batch_result_intent[i][0].decode("utf-8")
                result_ner = json.loads(batch_result_entity[i][0].decode("utf-8"))
                
                results_json.append({
                    "transcript": result_transcript,
                    "transcript_itn": result_transcript_itn,
                    "intent": result_intent,
                    "entities": result_ner,
                })
        except Exception as e:
            logger.error(f"Error processing inference results: {e}")
            return [{
                "transcript": "",
                "transcript_itn": "",
                "intent": "error_processing_results",
                "entities": [],
            }]
        
        return results_json
