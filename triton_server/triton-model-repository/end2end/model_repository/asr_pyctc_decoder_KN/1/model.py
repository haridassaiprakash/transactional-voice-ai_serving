from hotword_utils import hotword_to_fn
from pyctcdecode import build_ctcdecoder
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack
import torch
import json
import numpy as np
import os

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        base_path = os.path.join(args["model_repository"], args["model_version"])
        self.vocab = ['<unk>', '▁ಮ', '▁ಸ', 'ತ್', 'ಲ್', '▁ಕ', 'ಿದ', 'ಾರ', 'ಲ್ಲ', '▁ನ', 'ನ್', '▁ಅ', 'ಂದ', 'ಾಗ', 'ರು', 'ತ್ತ', '▁ಪ', '▁ಹ', 'ನ್ನ', '್ರ', 'ಿಸ', '▁ಬ', 'ಗಳ', '್ಯ', 'ಕ್', '▁ವ', 'ಗೆ', '್ದ', 'ಲ್ಲಿ', 'ರ್', 'ನ್ನು', 'ರಿ', '▁ಆ', '▁ತ', 'ೆಯ', '▁ಇ', '▁ಎ', 'ಿದ್ದ', 'ಿಯ', 'ುವ', 'ಿಕ', '್ಟ', '▁ಮಾ', 'ುತ್ತ', 'ಸ್', 'ದು', '▁ಪ್ರ', '▁ರ', '▁ಜ', 'ಾಗಿ', 'ಿದೆ', 'ಂತ', 'ಕ್ಕ', '▁ದ', 'ಿನ', 'ಂದು', 'ವು', 'ರೆ', '▁ಗ', '▁ಶ', 'ಾನ', 'ತ್ತು', 'ಂಡ', '▁ವಿ', '▁ನಿ', 'ಾಯ', '▁ಮಾಡ', 'ಾದ', 'ತಿ', '▁ಈ', 'ದಲ್ಲಿ', 'ಲು', 'ಗಳು', 'ಿಗೆ', '▁ಅವ', 'ಟ್ಟ', 'ಾರೆ', 'ಕ್ಷ', '▁ಉ', '್ಮ', '▁ನೀ', '▁ಚ', 'ದೆ', '▁ಸಂ', '▁ಒ', 'ಿಂದ', 'ಿತ', 'ಾಲ', '▁ಮತ್ತು', '▁ಭ', '▁ಯ', 'ಾಮ', 'ಕ್ಕೆ', 'ೇಕ', 'ತ್ರ', '▁ಮು', 'ಾವ', 'ನೆ', 'ಿಲ್ಲ', '▁ಸ್', 'ಿದ್ದಾರೆ', 'ಮ್ಮ', 'ಡಿ', 'ೊಳ', 'ಗಳನ್ನು', 'ಂಬ', '್ಳ', 'ಡೆ', 'ಷ್ಟ', 'ರುವ', 'ಚ್', 'ವಾಗಿ', '▁ಎಂದು', 'ಾಜ', 'ೇಶ', 'ಂಗ', 'ವನ್ನು', 'ುದ', 'ಿವ', 'ಾಗಿದೆ', 'ಪ್', '▁ಕೆ', 'ುದು', '▁ಕಾರ', 'ರಿಸ', 'ಬೇಕ', 'ಚ್ಚ', 'ೊಂಡ', 'ಾಸ', 'ೇಳ', 'ಾರಿ', '▁ಹೆ', '▁ಆದ', 'ಿದರು', '▁ಇದ', 'ತೆ', '▁ಹೊ', 'ಲೆ', '▁ಸಮ', '▁ಬೆ', 'ನಾ', 'ುತ್ತದೆ', 'ೋಗ', 'ೊಳ್ಳ', 'ಾಣ', 'ಧ್ಯ', 'ಾರ್', '▁ನೀಡ', 'ಿಸಿ', '▁ಬಿ', 'ವೆ', 'ನು', '▁ಸಾ', 'ರಿಗೆ', 'ತ್ಯ', '▁ಕು', 'ಪ್ಪ', 'ವಿ', 'ಂತೆ', 'ಕಾರ', '▁ಹಾಗ', '▁ಮೂ', 'ಡ್', 'ರಣ', '▁ಲ', 'ೆಯಲ್ಲಿ', 'ಬ್', 'ಗ್', '▁ಸಿ', '್ಣ', 'ವಾ', '▁ತಿ', 'ಟ್', '▁', '್', 'ಿ', 'ರ', 'ು', 'ದ', 'ಾ', 'ನ', 'ೆ', 'ತ', 'ಕ', 'ಲ', 'ಗ', 'ವ', 'ಸ', 'ಯ', 'ಮ', 'ಂ', 'ಳ', 'ಪ', 'ಡ', 'ಬ', 'ಹ', 'ೇ', 'ಟ', 'ಅ', 'ೂ', 'ೊ', 'ೀ', 'ಜ', 'ಚ', 'ಣ', 'ಶ', 'ೋ', 'ಷ', 'ಆ', 'ಧ', 'ಎ', 'ಇ', 'ಭ', 'ಥ', 'ೈ', 'ಈ', 'ಖ', 'ಉ', 'ಒ', 'ೃ', 'ಫ', 'ೌ', 'ಘ', 'ಠ', 'ಏ', 'ಐ', 'ಞ', 'ಓ', 'ಛ', '೦', 'ಊ', 'ಃ', 'ಢ', 'ೕ', 'ಔ', '೧', 'ಝ', '೨', '೯', 'ಋ', '೫', '೩', '೮', '೪', '೬', '೭', '಼', 'ಙ', 'ಱ', 'ೖ', 'ಽ', 'ೞ', 'ೄ', '\u0cbb', 'ಌ', 'ೠ']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="kn", base_path=base_path)
        self.hotword_weight=10 # TODO: Move all the static variables to the config.pbtxt file as parameters
        self.lm_path=None
        self.alpha = 1
        self.beta = 1
        if self.lm_path is not None:
            self.decoder = build_ctcdecoder(self.vocab, self.lm_path, alpha=self.alpha, beta=self.beta)
        else:
            self.decoder = build_ctcdecoder(self.vocab)
        
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSCRIPT")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "LOG_PROBS")
            print("in_0 :",in_0)
            in_1 = pb_utils.get_input_tensor_by_name(request, "HOTWORD_LIST")
            in_2 = pb_utils.get_input_tensor_by_name(request, "HOTWORD_WEIGHT")
            if in_1 is not None:
                hotword_list = in_1.as_numpy().tolist()
            
            if in_2 is not None:
                hotword_weight = from_dlpack(in_2.to_dlpack())
            
            logits = from_dlpack(in_0.to_dlpack())
            vocab_start_index = 1792
            logits_l = torch.cat([logits[:,:,vocab_start_index:vocab_start_index+256], logits[:,:,-1:]], dim=-1)
            logits_np = logits_l.detach().cpu().numpy()

            
            transcripts = []
            for i in range(len(logits)):
                # GET HOTWORDS LIST                
                hotword_l = self.static_hotwords_list 
                if in_1 is not None:
                    hotword_l += [hw.decode("UTF-8") for hw in hotword_list[i]]
                # GET HOTWORDS WEIGHT
                if in_2 is not None:                
                    hotword_w = hotword_weight[i].item()
                else:
                    hotword_w = self.hotword_weight
                transcript = self.decoder.decode(logits_np[i], hotwords=hotword_l, hotword_weight=hotword_w)
                transcripts.append(transcript.encode('utf-8'))
            out_numpy = np.array(transcripts).astype(self.output0_dtype)

            out_tensor_0 = pb_utils.Tensor("TRANSCRIPT", out_numpy)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses
