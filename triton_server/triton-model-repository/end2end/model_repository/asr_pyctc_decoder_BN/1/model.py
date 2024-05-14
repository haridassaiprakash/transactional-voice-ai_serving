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
        self.vocab = ['<unk>', 'য়', 'ার', '▁ক', '▁স', '▁ব', 'ের', '▁প', '্র', '্য', 'ান', '▁এ', '▁আ', '▁ম', '▁হ', '▁ন', '▁দ', '▁কর', '▁ত', '্ত', 'য়ে', 'েন', '▁অ', '▁জ', 'কে', 'িন', 'াল', 'ায়', 'তে', '▁প্র', 'িক', '▁শ', 'ছে', '▁র', '▁য', 'াম', 'টি', 'র্', 'বে', '▁বি', '▁গ', 'ড়', 'লে', '▁চ', 'লা', '▁নি', 'াস', '▁ভ', '▁ও', '্ব', 'তি', '▁উ', '▁পর', '্ট', 'াক', 'দের', '্ষ', '▁থ', 'িত', 'াজ', '▁করে', 'িল', 'ুর', 'াই', '▁এক', 'ুল', '▁দে', 'িয়ে', 'াত', '▁বা', '▁সম', 'ন্', '্থ', 'ির', '্যা', '▁ফ', '▁খ', '▁তা', 'য়া', '▁ছ', 'নে', 'রা', '▁ই', '▁আম', '▁হয়ে', 'েশ', 'বার', '▁না', 'ন্ত', '্প', 'োন', 'েকে', '▁জন', 'বা', 'ঙ্', 'ছেন', 'ক্ষ', '▁সে', '▁থেকে', 'িস', '▁তার', '▁হয়', '▁এই', 'য়ার', '▁ট', '▁১', 'েল', 'ভা', 'োগ', 'কার', 'দ্', 'িনি', '▁ল', 'ঙ্গ', '▁সা', '▁ঘ', 'চ্', 'টা', 'না', 'ক্ত', 'বি', 'নি', 'ধ্য', '▁জান', '▁আর', '▁পা', 'নের', '▁করা', '▁ধ', '▁অন', '▁পার', '্ম', '▁সং', 'ীর', '▁এব', '▁এবং', '▁২', '▁ড', '▁মা', 'তা', '▁নে', 'ীয়', '▁যে', 'দ্ধ', 'স্থ', 'িশ', 'রে', '▁যা', '▁উপ', 'ুন', 'ষ্', '▁ব্য', '▁তিনি', '▁পরি', 'াপ', 'ানে', '▁হয়েছে', '▁জন্য', '▁দু', '▁কি', '▁নিয়ে', 'দেশ', '▁কার', 'ছিল', '▁', 'া', 'ে', 'র', '্', 'ি', 'ন', 'ক', 'য', 'ব', 'ত', 'স', 'ম', 'ল', '়', 'প', 'দ', 'ু', 'হ', 'ট', 'জ', 'ো', 'শ', 'গ', 'ছ', 'এ', 'ই', 'আ', 'ী', 'চ', 'থ', 'ড', 'ও', 'ভ', 'ষ', 'ধ', 'খ', 'অ', 'ং', 'উ', 'ণ', 'ফ', 'ঠ', '১', 'ৃ', 'ঁ', 'ূ', 'ঘ', 'ঙ', '২', '০', 'ঞ', '৫', 'ৈ', '৩', 'ৌ', '৯', '৪', 'ৎ', '৬', 'ঝ', '৮', '৭', 'ঢ', 'ঃ', 'ঐ', 'ঈ', '৷', 'ঋ', 'ঊ', 'ঔ', 'ৰ', 'ৗ', 'ৱ', '\u09e4', '৳', '\u09a9', '\u09b3', '\u09b4', '\u09e5', 'ঌ', '\u0991', 'ৄ', 'ৠ', '\u09b1', 'ঽ', '৴', '৻', 'ৡ', '৲', '\u0984', '৵']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="bn", base_path=base_path)
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
            vocab_start_index = 256
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
