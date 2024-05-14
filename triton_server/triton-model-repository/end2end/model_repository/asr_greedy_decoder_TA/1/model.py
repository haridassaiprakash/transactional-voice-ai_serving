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
        self.vocab = ['<unk>', '்க', '்த', 'ம்', 'ல்', '▁ப', 'ன்', 'க்க', '▁க', '்ட', '▁வ', '▁ம', 'த்த', '்ப', '▁அ', 'ும்', '▁த', '▁ச', 'ள்', 'ிய', '▁இ', 'ர்', 'ரு', 'ந்த', 'ப்ப', 'து', 'ட்ட', '▁ந', 'ில்', '▁எ', 'ங்க', 'ைய', 'ாக', 'ிற', 'ின்', 'டு', '▁உ', 'க்கு', 'ற்', 'ிர', 'ிக', 'று', 'ண்ட', 'ார', '்ச', 'ள்ள', 'த்து', '▁மு', 'ில', '▁ஆ', 'ான', 'ற்ற', '▁செ', 'டி', 'ர்க', 'ார்', 'லை', '▁வி', '▁ஒ', '▁என்', 'ட்டு', 'ந்து', 'வி', 'ால்', 'ளை', 'ப்', 'ரி', 'தி', '▁கு', 'ிரு', '▁இரு', 'வு', '▁அவ', '▁கொ', '▁போ', 'ல்ல', 'க்', '▁செய', 'ச்ச', 'ின', '▁கா', '▁அத', 'த்', 'ான்', 'மை', '▁பெ', '▁மா', 'ளு', '▁வே', 'றி', 'த்தில்', '▁இந்த', '▁ஒரு', 'க்கும்', 'னை', 'ப்பு', 'ையில்', 'ாய', 'ங்கள்', '▁தொ', 'டை', 'ற்க', 'ர்கள்', 'ம்ப', 'ன்ற', '▁ஏ', 'ரா', 'ுவ', 'ஸ்', 'ண்', 'ால', 'ிக்க', 'டிய', 'னர்', 'ண்டு', '▁வெ', 'ாவ', 'ிறது', '▁பு', 'ாத', 'கள்', 'மாக', 'ாள', '▁கூ', 'மி', 'ச்', 'ன்ன', 'றை', 'வே', '்கள்', '▁உள்ள', 'கு', 'ப்பட்ட', 'ாம்', '▁என', '▁மற்ற', 'ற்ப', 'ங்கள', '▁தெ', 'ழு', '▁பிர', '▁பொ', 'த்தை', 'ரிய', 'டுத்த', 'மான', '▁பா', '▁தே', '▁நட', 'ரை', 'ளுக்கு', 'வும்', 'ையும்', '▁இத', 'ரச', 'ட்ச', 'திய', '்த்த', 'ின்ற', '▁மற்றும்', 'ிட', 'ாம', 'கள', '▁நீ', 'ரும்', 'வர்', '▁மே', 'வை', '▁வா', 'ற்று', '▁நா', 'வத', '▁வழ', 'மு', 'ண்டும்', 'டும்', '▁ஜ', 'வில்', '▁ர', 'டன்', '▁செய்', 'ண்ண', '▁சு', 'ன்று', '▁தொட', 'ர்கள', '▁வரு', '▁அரச', 'னால்', '்', '▁', 'க', 'ு', 'த', 'ி', 'ப', 'ர', 'ம', 'ட', 'ா', 'வ', 'ல', 'ன', 'ை', 'ள', 'ய', 'ற', 'ச', 'ந', 'அ', 'ே', 'ண', 'இ', 'ெ', 'ோ', 'எ', 'ங', 'ொ', 'ழ', 'உ', 'ீ', 'ூ', 'ஆ', 'ஒ', 'ஸ', 'ஜ', 'ஏ', 'ஷ', 'ஞ', 'ஊ', 'ஹ', 'ஓ', 'ஐ', 'ஈ', 'ஃ', 'ௌ', 'ஶ', 'ஔ', 'ஂ', '௧', '௦', '௨', '௭', '௫', '௩', '௮', '௬', '௪', '௯', 'ௐ', 'ௗ', '\u0b91', '\u0b80', '\u0b9b', '\u0ba2', '\u0ba6']
        
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
