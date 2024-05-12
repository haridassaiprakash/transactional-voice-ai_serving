import numpy as np
import json
from swig_decoders import map_batch

import triton_python_backend_utils as pb_utils
import multiprocessing

class TritonPythonModel:

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        parameters = model_config["parameters"]
        self.subsampling_rate = 2
        self.blank_id = 0
        for li in parameters.items():
            key, value = li
            value = value['string_value']
            if key == "subsampling_rate":
                self.subsampling_rate = int(value)
            elif key == "blank_id":
                self.blank_id = int(value)
        self.vocab = ['<unk>', '▁ಮ', '▁ಸ', 'ತ್', 'ಲ್', '▁ಕ', 'ಿದ', 'ಾರ', 'ಲ್ಲ', '▁ನ', 'ನ್', '▁ಅ', 'ಂದ', 'ಾಗ', 'ರು', 'ತ್ತ', '▁ಪ', '▁ಹ', 'ನ್ನ', '್ರ', 'ಿಸ', '▁ಬ', 'ಗಳ', '್ಯ', 'ಕ್', '▁ವ', 'ಗೆ', '್ದ', 'ಲ್ಲಿ', 'ರ್', 'ನ್ನು', 'ರಿ', '▁ಆ', '▁ತ', 'ೆಯ', '▁ಇ', '▁ಎ', 'ಿದ್ದ', 'ಿಯ', 'ುವ', 'ಿಕ', '್ಟ', '▁ಮಾ', 'ುತ್ತ', 'ಸ್', 'ದು', '▁ಪ್ರ', '▁ರ', '▁ಜ', 'ಾಗಿ', 'ಿದೆ', 'ಂತ', 'ಕ್ಕ', '▁ದ', 'ಿನ', 'ಂದು', 'ವು', 'ರೆ', '▁ಗ', '▁ಶ', 'ಾನ', 'ತ್ತು', 'ಂಡ', '▁ವಿ', '▁ನಿ', 'ಾಯ', '▁ಮಾಡ', 'ಾದ', 'ತಿ', '▁ಈ', 'ದಲ್ಲಿ', 'ಲು', 'ಗಳು', 'ಿಗೆ', '▁ಅವ', 'ಟ್ಟ', 'ಾರೆ', 'ಕ್ಷ', '▁ಉ', '್ಮ', '▁ನೀ', '▁ಚ', 'ದೆ', '▁ಸಂ', '▁ಒ', 'ಿಂದ', 'ಿತ', 'ಾಲ', '▁ಮತ್ತು', '▁ಭ', '▁ಯ', 'ಾಮ', 'ಕ್ಕೆ', 'ೇಕ', 'ತ್ರ', '▁ಮು', 'ಾವ', 'ನೆ', 'ಿಲ್ಲ', '▁ಸ್', 'ಿದ್ದಾರೆ', 'ಮ್ಮ', 'ಡಿ', 'ೊಳ', 'ಗಳನ್ನು', 'ಂಬ', '್ಳ', 'ಡೆ', 'ಷ್ಟ', 'ರುವ', 'ಚ್', 'ವಾಗಿ', '▁ಎಂದು', 'ಾಜ', 'ೇಶ', 'ಂಗ', 'ವನ್ನು', 'ುದ', 'ಿವ', 'ಾಗಿದೆ', 'ಪ್', '▁ಕೆ', 'ುದು', '▁ಕಾರ', 'ರಿಸ', 'ಬೇಕ', 'ಚ್ಚ', 'ೊಂಡ', 'ಾಸ', 'ೇಳ', 'ಾರಿ', '▁ಹೆ', '▁ಆದ', 'ಿದರು', '▁ಇದ', 'ತೆ', '▁ಹೊ', 'ಲೆ', '▁ಸಮ', '▁ಬೆ', 'ನಾ', 'ುತ್ತದೆ', 'ೋಗ', 'ೊಳ್ಳ', 'ಾಣ', 'ಧ್ಯ', 'ಾರ್', '▁ನೀಡ', 'ಿಸಿ', '▁ಬಿ', 'ವೆ', 'ನು', '▁ಸಾ', 'ರಿಗೆ', 'ತ್ಯ', '▁ಕು', 'ಪ್ಪ', 'ವಿ', 'ಂತೆ', 'ಕಾರ', '▁ಹಾಗ', '▁ಮೂ', 'ಡ್', 'ರಣ', '▁ಲ', 'ೆಯಲ್ಲಿ', 'ಬ್', 'ಗ್', '▁ಸಿ', '್ಣ', 'ವಾ', '▁ತಿ', 'ಟ್', '▁', '್', 'ಿ', 'ರ', 'ು', 'ದ', 'ಾ', 'ನ', 'ೆ', 'ತ', 'ಕ', 'ಲ', 'ಗ', 'ವ', 'ಸ', 'ಯ', 'ಮ', 'ಂ', 'ಳ', 'ಪ', 'ಡ', 'ಬ', 'ಹ', 'ೇ', 'ಟ', 'ಅ', 'ೂ', 'ೊ', 'ೀ', 'ಜ', 'ಚ', 'ಣ', 'ಶ', 'ೋ', 'ಷ', 'ಆ', 'ಧ', 'ಎ', 'ಇ', 'ಭ', 'ಥ', 'ೈ', 'ಈ', 'ಖ', 'ಉ', 'ಒ', 'ೃ', 'ಫ', 'ೌ', 'ಘ', 'ಠ', 'ಏ', 'ಐ', 'ಞ', 'ಓ', 'ಛ', '೦', 'ಊ', 'ಃ', 'ಢ', 'ೕ', 'ಔ', '೧', 'ಝ', '೨', '೯', 'ಋ', '೫', '೩', '೮', '೪', '೬', '೭', '಼', 'ಙ', 'ಱ', 'ೖ', 'ಽ', 'ೞ', 'ೄ', '\u0cbb', 'ಌ', 'ೠ']
        
        if self.blank_id == -1:
            self.blank_id = len(self.vocab)
        self.num_processes = multiprocessing.cpu_count()
        if args["model_instance_kind"] == "GPU":
            print("GPU GREEDY DECODER IS NOT SUPPORTED!")
        
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSCRIPT")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        """
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        batch_transcript = []
        request_batch_size = []
        responses = []
        for request in requests:
            # B X T
            in_0 = pb_utils.get_input_tensor_by_name(request, "TRANSCRIPT_ID")
            transcript_id = in_0.as_numpy().tolist()
            cur_batch_size = len(transcript_id)
            request_batch_size.append(cur_batch_size)
            # B X 1
            in_1 = pb_utils.get_input_tensor_by_name(request, "NUM_TIME_STEPS")
            timesteps = in_1.as_numpy()
            for i in range(cur_batch_size):
                cur_len = (timesteps[i][0] + 1) // self.subsampling_rate
                batch_transcript.append(transcript_id[i][0:cur_len])

        num_processes = min(self.num_processes, len(batch_transcript))
        res_sents = map_batch(batch_transcript, self.vocab, num_processes, True, self.blank_id)
        start = 0
        for b in request_batch_size:
            sent = res_sents[start:start+b]
            sent = np.array([s.replace("▁", " ").strip() for s in sent])
            out_tensor_0 = pb_utils.Tensor("TRANSCRIPT", sent.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
            start = start + b
        return responses
