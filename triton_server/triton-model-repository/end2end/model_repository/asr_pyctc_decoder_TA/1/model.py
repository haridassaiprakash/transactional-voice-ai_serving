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
        self.vocab = ['<unk>', '்க', '்த', 'ம்', 'ல்', '▁ப', 'ன்', 'க்க', '▁க', '்ட', '▁வ', '▁ம', 'த்த', '்ப', '▁அ', 'ும்', '▁த', '▁ச', 'ள்', 'ிய', '▁இ', 'ர்', 'ரு', 'ந்த', 'ப்ப', 'து', 'ட்ட', '▁ந', 'ில்', '▁எ', 'ங்க', 'ைய', 'ாக', 'ிற', 'ின்', 'டு', '▁உ', 'க்கு', 'ற்', 'ிர', 'ிக', 'று', 'ண்ட', 'ார', '்ச', 'ள்ள', 'த்து', '▁மு', 'ில', '▁ஆ', 'ான', 'ற்ற', '▁செ', 'டி', 'ர்க', 'ார்', 'லை', '▁வி', '▁ஒ', '▁என்', 'ட்டு', 'ந்து', 'வி', 'ால்', 'ளை', 'ப்', 'ரி', 'தி', '▁கு', 'ிரு', '▁இரு', 'வு', '▁அவ', '▁கொ', '▁போ', 'ல்ல', 'க்', '▁செய', 'ச்ச', 'ின', '▁கா', '▁அத', 'த்', 'ான்', 'மை', '▁பெ', '▁மா', 'ளு', '▁வே', 'றி', 'த்தில்', '▁இந்த', '▁ஒரு', 'க்கும்', 'னை', 'ப்பு', 'ையில்', 'ாய', 'ங்கள்', '▁தொ', 'டை', 'ற்க', 'ர்கள்', 'ம்ப', 'ன்ற', '▁ஏ', 'ரா', 'ுவ', 'ஸ்', 'ண்', 'ால', 'ிக்க', 'டிய', 'னர்', 'ண்டு', '▁வெ', 'ாவ', 'ிறது', '▁பு', 'ாத', 'கள்', 'மாக', 'ாள', '▁கூ', 'மி', 'ச்', 'ன்ன', 'றை', 'வே', '்கள்', '▁உள்ள', 'கு', 'ப்பட்ட', 'ாம்', '▁என', '▁மற்ற', 'ற்ப', 'ங்கள', '▁தெ', 'ழு', '▁பிர', '▁பொ', 'த்தை', 'ரிய', 'டுத்த', 'மான', '▁பா', '▁தே', '▁நட', 'ரை', 'ளுக்கு', 'வும்', 'ையும்', '▁இத', 'ரச', 'ட்ச', 'திய', '்த்த', 'ின்ற', '▁மற்றும்', 'ிட', 'ாம', 'கள', '▁நீ', 'ரும்', 'வர்', '▁மே', 'வை', '▁வா', 'ற்று', '▁நா', 'வத', '▁வழ', 'மு', 'ண்டும்', 'டும்', '▁ஜ', 'வில்', '▁ர', 'டன்', '▁செய்', 'ண்ண', '▁சு', 'ன்று', '▁தொட', 'ர்கள', '▁வரு', '▁அரச', 'னால்', '்', '▁', 'க', 'ு', 'த', 'ி', 'ப', 'ர', 'ம', 'ட', 'ா', 'வ', 'ல', 'ன', 'ை', 'ள', 'ய', 'ற', 'ச', 'ந', 'அ', 'ே', 'ண', 'இ', 'ெ', 'ோ', 'எ', 'ங', 'ொ', 'ழ', 'உ', 'ீ', 'ூ', 'ஆ', 'ஒ', 'ஸ', 'ஜ', 'ஏ', 'ஷ', 'ஞ', 'ஊ', 'ஹ', 'ஓ', 'ஐ', 'ஈ', 'ஃ', 'ௌ', 'ஶ', 'ஔ', 'ஂ', '௧', '௦', '௨', '௭', '௫', '௩', '௮', '௬', '௪', '௯', 'ௐ', 'ௗ', '\u0b91', '\u0b80', '\u0b9b', '\u0ba2', '\u0ba6']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="ta", base_path=base_path)
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
            in_1 = pb_utils.get_input_tensor_by_name(request, "HOTWORD_LIST")
            in_2 = pb_utils.get_input_tensor_by_name(request, "HOTWORD_WEIGHT")
            if in_1 is not None:
                hotword_list = in_1.as_numpy().tolist()
            
            if in_2 is not None:
                hotword_weight = from_dlpack(in_2.to_dlpack())
            
            logits = from_dlpack(in_0.to_dlpack())
            vocab_start_index = 4864
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
