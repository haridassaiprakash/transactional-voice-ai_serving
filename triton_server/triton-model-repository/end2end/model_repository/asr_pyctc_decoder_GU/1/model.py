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
        self.vocab = ['<unk>', '▁ક', 'મા', '▁પ', '▁સ', 'વા', 'ને', 'ના', '▁ત', '્ય', '▁આ', '▁છ', '▁જ', 'માં', '▁અ', '્ર', 'ાર', '▁છે', '▁હ', 'ની', '▁મ', '▁ર', '▁વ', '▁બ', 'ું', '▁કર', 'તા', '▁તે', '▁દ', 'ર્', '▁ન', 'લા', '▁એ', '▁શ', 'થી', '▁અને', '▁મા', '▁ગ', '▁કે', '▁લ', 'રા', '્ટ', '્યા', 'નો', '▁આવ', 'રી', '▁હત', 'યા', '▁પ્ર', '▁સા', '▁થ', '▁ભ', 'ટે', 'િક', '▁પર', '▁ચ', '▁ખ', 'હે', 'સ્', '▁વિ', '▁ફ', '▁ઉ', 'લી', 'ક્', 'ન્', 'ાય', 'ાવ', '▁કો', '▁પણ', 'મે', '▁જે', 'ંગ', 'કો', '▁માટે', '▁રા', 'ંત', 'વામાં', 'નું', 'શે', 'િત', '▁કરી', '▁એક', 'વી', '▁પા', 'રો', '▁જો', '▁હો', '▁સ્', '્યો', 'રે', 'હી', 'યો', '્યું', '▁પો', 'લ્', 'કે', 'િયા', 'કા', 'ણી', '▁વા', 'ડી', '▁તેમ', 'ારે', '▁ટ', '▁આપ', 'ામ', 'તી', 'થે', '▁ઘ', 'ેશ', '▁બા', '▁સં', 'ત્', '▁કાર', '▁મો', '▁સાથે', '▁ધ', 'ાન', 'લે', '▁હતી', '▁ડ', 'કાર', '▁સુ', 'ણે', '▁લો', '▁ના', 'તિ', '્રી', '▁સમ', '▁લા', 'ભા', '▁વધ', 'જી', 'વે', 'ડા', 'તે', 'ંધ', '▁મુ', '▁તો', '▁બે', 'ક્ષ', 'ાલ', '▁રહ', 'ટી', '▁જા', 'ંદ', 'સે', '▁મળ', '▁કરવા', 'ત્ર', 'ારી', '▁હતા', '▁ઓ', 'ાસ', 'િવ', '▁હતો', '▁ઉપ', 'રૂ', 'સા', '▁નિ', 'કી', '▁નથી', '્યારે', '▁મહ', 'દી', '▁આવે', '▁તમા', '▁આવી', 'તો', 'જા', 'સ્ટ', 'સી', '▁', 'ા', 'ે', 'ર', 'ન', 'ી', '્', 'ક', 'મ', 'ત', 'વ', 'ો', 'ં', 'સ', 'પ', 'ય', 'િ', 'લ', 'જ', 'ુ', 'હ', 'ટ', 'દ', 'ગ', 'છ', 'આ', 'થ', 'બ', 'શ', 'અ', 'ણ', 'ડ', 'ધ', 'એ', 'ચ', 'ખ', 'ભ', 'ૂ', 'ળ', 'ફ', 'ઈ', 'ઓ', 'ષ', 'ઇ', 'ઉ', 'ઘ', 'ઝ', 'ઠ', '૦', '૧', 'ૃ', 'ૈ', '૨', 'ૌ', 'ઢ', '૫', '૩', '૪', '૬', '૯', '૮', '૭', 'ૉ', 'ઃ', 'ઊ', 'ઞ', 'ૅ', 'ઑ', 'ઔ', 'ઋ', 'ઐ', 'ઍ', 'ઙ', 'ઁ', '઼', 'ૐ', 'ૠ', 'ૢ', 'ૄ', 'ઽ', '\u0aa9', '\u0ae4', '૱', '\u0ab1']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="gu", base_path=base_path)
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
            vocab_start_index = 1280
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
