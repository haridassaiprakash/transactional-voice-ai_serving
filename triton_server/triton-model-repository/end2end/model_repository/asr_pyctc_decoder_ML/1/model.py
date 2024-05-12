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
        self.vocab = ['<unk>', 'ന്', 'ക്', 'ത്', '▁പ', 'ന്ന', 'ക്ക', 'തി', '▁ക', '്ട', 'ും', '▁സ', '▁വ', 'യി', '▁അ', '▁മ', '▁ന', 'ുന്ന', 'ച്', '്പ', 'ങ്', '്ര', '്യ', 'ാണ', '▁എ', 'ത്തി', 'രി', 'ട്ട', 'ത്ത', 'പ്പ', 'ങ്ങ', 'സ്', 'ില', 'റ്', 'ിയ', 'ച്ച', 'മാ', 'ാണ്', '▁ത', 'ണ്ട', '▁ഇ', 'ുക', 'ടെ', '▁ച', '▁ആ', 'രു', '്ല', 'ിക്ക', '▁പ്ര', 'റ്റ', '▁വി', 'ിൽ', 'ുന്നു', 'ാന', 'ായി', '്ള', 'ള്ള', 'റെ', 'ഞ്', 'ിച്ച', 'ാര', '▁ര', '▁ഒ', '▁ജ', '▁ഉ', 'െയ', '▁ബ', 'ിക', 'ക്ക്', 'ുടെ', 'ടു', '▁നി', 'ന്റെ', 'ന്ന്', 'ായ', 'ങ്ങള', 'ക്ഷ', 'വി', 'ല്ല', 'ുള്ള', 'ത്ര', '▁സ്', '▁ശ', 'ദ്', 'ഞ്ഞ', '▁പി', 'റി', 'ാൻ', 'ുമ', 'െന്ന', 'ങ്ങൾ', '▁എന്ന', 'ാല', 'രുന്നു', 'യും', 'ിന', 'രിക്ക', '▁സം', 'മായി', 'ടി', 'പ്പെ', 'ാർ', 'ണ്ട്', '▁കു', '▁ല', 'യിൽ', '▁ഒരു', 'ില്ല', 'ങ്ക', 'ാവ', 'ദേ', '▁ചെയ', 'ുന്നത്', '്മ', 'യില', 'ത്തിൽ', '▁മു', '▁മാ', 'വർ', 'ണം', '▁ഭ', 'ാക്ക', '▁നട', 'തു', '▁യ', 'ോഗ', 'േഷ', 'മായ', 'ിവ', 'ാം', '▁പറ', 'മ്മ', '▁ഗ', '▁പോ', '▁ഡ', 'ാമ', 'ത്തില', 'ുവ', 'തിന', 'ത്യ', '▁ദ', 'വും', '▁പു', 'ത്ത്', 'സി', 'ച്ച്', '▁കോ', 'െന്ന്', 'ത്തെ', '▁സി', '▁കൊ', 'വാ', 'ുകള', '▁അവ', 'രെ', 'ാൽ', '▁ഈ', '▁കേ', 'സ്ഥ', 'ദ്യ', '▁തു', 'ന്ത', 'യാണ്', '▁ഫ', 'ായിരുന്നു', '്', '▁', 'ി', 'ക', 'ന', 'ു', 'ത', 'ാ', 'യ', 'ര', 'ട', 'പ', 'െ', 'മ', 'വ', 'ം', 'ല', 'സ', 'റ', 'ച', 'ണ', 'ള', 'ോ', 'ങ', 'േ', 'ർ', 'ൽ', 'അ', 'ദ', 'ീ', 'എ', 'ഷ', 'ശ', 'ജ', 'ൻ', 'ഗ', 'ൾ', 'ധ', 'ഞ', 'ൂ', 'ഇ', 'ബ', 'ആ', 'ഹ', 'ൊ', 'ഭ', 'ഡ', 'ഴ', 'ഒ', 'ഉ', 'ഥ', 'ൈ', 'ഫ', 'ൃ', 'ഖ', 'ഈ', 'ഏ', 'ഘ', 'ൺ', 'ഓ', 'ൗ', 'ഐ', 'ഠ', 'ഛ', 'ഊ', 'ഔ', 'ൌ', 'ഃ', 'ഢ', 'ഋ', '൪', 'ഝ', '൯', '൦', 'ഌ', 'ൿ', '഼', 'ൎ', 'ൡ', '൧', '൨', '൫', '൬', '൩', '൭', 'ഽ', '൮', 'ഺ', 'ൟ', 'ഩ', 'ൠ']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="ml", base_path=base_path)
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
            vocab_start_index = 2560
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