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
        self.vocab = ['<unk>', 'या', '्या', '▁क', '▁आ', '▁प', '▁स', '▁म', 'ार', '्र', '▁त', '▁अ', 'ला', '▁व', '▁ह', 'ना', 'ात', '▁द', 'हे', 'र्', '▁या', '▁आहे', 'ले', 'ां', '्य', 'च्या', '▁ज', '▁न', 'ही', '▁र', 'ली', 'ती', 'ून', '▁ब', 'ल्या', '▁त्या', 'चा', 'ने', 'िक', '▁अस', '▁श', '▁कर', 'वा', 'ता', 'ची', 'ण्या', 'चे', '▁प्र', '▁ग', 'क्', '▁हो', 'नी', '▁का', '्ह', '▁के', 'वर', 'स्', 'ंत', 'सा', '▁भ', '▁आण', '▁घ', 'ाज', '▁उ', 'ते', '▁आणि', '▁वि', '▁ना', 'ित', 'ाव', '▁ए', '▁यां', 'ेत', '▁य', 'ंद', 'ील', '▁दि', 'ान', '▁झ', 'ध्य', '▁ल', 'ठी', 'त्', '▁पा', 'क्ष', '▁नि', '▁च', 'का', '▁सं', '▁झा', '▁ख', 'री', '▁वा', '▁मा', 'णा', '्ट', 'ळे', 'ास', 'साठी', '▁फ', '▁सु', 'मु', 'हि', 'कार', 'ध्ये', '▁त्यां', '▁दे', '▁मु', '▁स्', 'णार', 'मा', 'णी', 'रा', '▁ला', '▁को', 'ण्यात', '▁नाही', 'मध्ये', 'रो', '्यां', 'ाय', '▁ट', '▁आहेत', 'शी', 'कर', 'मुळे', 'ंग', 'णे', '▁म्ह', '▁जा', '▁हे', '▁रा', '्री', '▁ठ', '▁एक', 'लं', '▁ते', 'वि', 'वे', 'द्', '▁इ', 'ष्ट', 'डे', '▁पर', '▁सा', 'से', '▁आप', 'ळी', '▁तर', '▁पु', '▁यांनी', '▁राज', 'ारी', 'ल्', '▁ये', '▁मह', 'र्व', '्हा', '▁कार', '▁मि', '▁मो', '▁', 'ा', '्', 'र', 'े', 'त', 'य', 'ी', 'क', 'ल', 'न', 'स', 'व', 'ि', 'ह', 'ं', 'म', 'च', 'प', 'ण', 'ो', 'द', 'आ', 'ु', 'ज', 'ग', 'श', 'अ', 'ट', 'ू', 'ब', 'ड', 'ध', 'ळ', 'ष', 'भ', 'ख', 'ठ', 'घ', 'थ', 'उ', 'झ', 'ए', 'फ', 'ई', 'ढ', 'ॉ', 'इ', 'ृ', 'ै', 'ऊ', '१', 'ऱ', '०', 'ॅ', '२', 'ौ', 'ँ', 'ओ', '५', 'छ', '३', '४', '९', 'ऑ', '६', '८', '७', 'ञ', 'ः', '़', 'ऐ', 'औ', 'ऋ', 'ऍ', 'ॲ', 'ङ', 'ऽ', 'ॆ', 'ॊ', 'ॄ', 'ॐ', 'ऴ', 'ॕ', 'ऩ', 'ऎ', 'ॠ', '॑', 'ऒ', '॰', '॓', '॔', 'ॽ', 'ऌ', 'ऺ']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="mr", base_path=base_path)
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
            vocab_start_index = 2816
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
