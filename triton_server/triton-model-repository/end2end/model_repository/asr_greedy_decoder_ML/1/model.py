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
        self.vocab = ['<unk>', 'ന്', 'ക്', 'ത്', '▁പ', 'ന്ന', 'ക്ക', 'തി', '▁ക', '്ട', 'ും', '▁സ', '▁വ', 'യി', '▁അ', '▁മ', '▁ന', 'ുന്ന', 'ച്', '്പ', 'ങ്', '്ര', '്യ', 'ാണ', '▁എ', 'ത്തി', 'രി', 'ട്ട', 'ത്ത', 'പ്പ', 'ങ്ങ', 'സ്', 'ില', 'റ്', 'ിയ', 'ച്ച', 'മാ', 'ാണ്', '▁ത', 'ണ്ട', '▁ഇ', 'ുക', 'ടെ', '▁ച', '▁ആ', 'രു', '്ല', 'ിക്ക', '▁പ്ര', 'റ്റ', '▁വി', 'ിൽ', 'ുന്നു', 'ാന', 'ായി', '്ള', 'ള്ള', 'റെ', 'ഞ്', 'ിച്ച', 'ാര', '▁ര', '▁ഒ', '▁ജ', '▁ഉ', 'െയ', '▁ബ', 'ിക', 'ക്ക്', 'ുടെ', 'ടു', '▁നി', 'ന്റെ', 'ന്ന്', 'ായ', 'ങ്ങള', 'ക്ഷ', 'വി', 'ല്ല', 'ുള്ള', 'ത്ര', '▁സ്', '▁ശ', 'ദ്', 'ഞ്ഞ', '▁പി', 'റി', 'ാൻ', 'ുമ', 'െന്ന', 'ങ്ങൾ', '▁എന്ന', 'ാല', 'രുന്നു', 'യും', 'ിന', 'രിക്ക', '▁സം', 'മായി', 'ടി', 'പ്പെ', 'ാർ', 'ണ്ട്', '▁കു', '▁ല', 'യിൽ', '▁ഒരു', 'ില്ല', 'ങ്ക', 'ാവ', 'ദേ', '▁ചെയ', 'ുന്നത്', '്മ', 'യില', 'ത്തിൽ', '▁മു', '▁മാ', 'വർ', 'ണം', '▁ഭ', 'ാക്ക', '▁നട', 'തു', '▁യ', 'ോഗ', 'േഷ', 'മായ', 'ിവ', 'ാം', '▁പറ', 'മ്മ', '▁ഗ', '▁പോ', '▁ഡ', 'ാമ', 'ത്തില', 'ുവ', 'തിന', 'ത്യ', '▁ദ', 'വും', '▁പു', 'ത്ത്', 'സി', 'ച്ച്', '▁കോ', 'െന്ന്', 'ത്തെ', '▁സി', '▁കൊ', 'വാ', 'ുകള', '▁അവ', 'രെ', 'ാൽ', '▁ഈ', '▁കേ', 'സ്ഥ', 'ദ്യ', '▁തു', 'ന്ത', 'യാണ്', '▁ഫ', 'ായിരുന്നു', '്', '▁', 'ി', 'ക', 'ന', 'ു', 'ത', 'ാ', 'യ', 'ര', 'ട', 'പ', 'െ', 'മ', 'വ', 'ം', 'ല', 'സ', 'റ', 'ച', 'ണ', 'ള', 'ോ', 'ങ', 'േ', 'ർ', 'ൽ', 'അ', 'ദ', 'ീ', 'എ', 'ഷ', 'ശ', 'ജ', 'ൻ', 'ഗ', 'ൾ', 'ധ', 'ഞ', 'ൂ', 'ഇ', 'ബ', 'ആ', 'ഹ', 'ൊ', 'ഭ', 'ഡ', 'ഴ', 'ഒ', 'ഉ', 'ഥ', 'ൈ', 'ഫ', 'ൃ', 'ഖ', 'ഈ', 'ഏ', 'ഘ', 'ൺ', 'ഓ', 'ൗ', 'ഐ', 'ഠ', 'ഛ', 'ഊ', 'ഔ', 'ൌ', 'ഃ', 'ഢ', 'ഋ', '൪', 'ഝ', '൯', '൦', 'ഌ', 'ൿ', '഼', 'ൎ', 'ൡ', '൧', '൨', '൫', '൬', '൩', '൭', 'ഽ', '൮', 'ഺ', 'ൟ', 'ഩ', 'ൠ']
        
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
