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
        self.vocab = ['<unk>', '▁କ', '▁ସ', '▁ପ', 'ାର', '▁ବ', '୍ର', '୍ତ', 'ରେ', '୍ୟ', '▁ମ', 'ଙ୍', 'ିବ', '▁ଏ', 'ଙ୍କ', '▁କର', 'ାନ', '▁ନ', '▁ଅ', '▁ଦ', '▁ହ', 'ାଇ', 'ିଲ', '▁ଆ', 'ର୍', 'ନ୍ତ', 'ହି', '▁ର', 'ିକ', 'ିବା', '▁ଜ', 'ଛି', 'କୁ', '▁ପ୍ର', '▁ତ', 'ଥିଲ', '▁ଗ', '▁ବି', '▁ଭ', 'ନ୍', '▁ଯ', 'ିତ', 'ନ୍ତି', '▁କରି', '▁ସେ', 'ରୁ', '▁ଲ', 'େବ', '୍ଷ', '▁ଓ', 'ମ୍', '▁ଶ', 'ନା', '▁ଚ', 'ଛନ୍ତି', 'ଟି', 'ହା', 'ୋଇ', '▁ଉ', 'ାରେ', '▁ହୋଇ', 'ଡ଼', 'ଥିବା', '▁ନି', 'ଲି', 'ଷ୍', 'ାମ', 'ଥିଲେ', 'େଇ', 'ଣ୍', 'ତି', '୍ୱ', '୍ଥ', 'ରି', 'ଙ୍କୁ', 'ାସ', 'କ୍ଷ', 'ାବ', 'ମାନ', 'ଧ୍ୟ', '▁ପାଇ', 'ନ୍ଦ', '▁ଏହି', 'ାତ', '▁ରା', '୍ୟା', '▁ସମ', 'ସି', 'ାଯ', '▁ଅନ', 'ଥିଲା', '▁ତା', 'େଶ', 'ୋଗ', '୍ଚ', 'ବା', '▁୧', '▁ମଧ୍ୟ', '▁ଖ', '▁ଘ', '▁ପାଇଁ', 'ିନ', 'ଡି', '୍ରୀ', '▁କି', 'ାଗ', '▁ଟ', '୍ଲ', 'ୁର', 'ଦ୍', 'ାପ', '▁ଜଣ', 'କାର', '▁୨', '▁ଏବ', 'ାଳ', 'ାୟ', 'କ୍ତ', '▁ଫ', '▁କରାଯ', 'ାର୍', 'ଙ୍ଗ', 'ୋଲି', 'େଳ', 'ସ୍ତ', '▁ସଂ', 'ାରୁ', '▁ହେ', 'ଷ୍ଟ', '▁ରାଜ', 'ଯ୍ୟ', 'ୋକ', '▁ଏହା', '▁ମୁ', 'ତା', '▁ଏକ', 'ାଲ', 'ଭି', 'ଳି', 'ାଣ', '▁ମି', '▁ସହ', '▁ସୁ', '▁ହେବ', '▁ପର', '▁କରିବା', 'ାଉ', 'ଡ଼ି', 'ାରି', '▁କେ', '▁ଉପ', 'ଣ୍ଡ', 'ଥା', 'ସ୍ଥ', 'କ୍ର', 'ାନ୍ତ', 'ଦ୍ଧ', 'ଲେ', 'ୀୟ', 'ଞ୍ଚ', '▁ଏବଂ', '▁ଯେ', 'ୃତ', '▁ଧ', '▁ପରେ', '▁ସମ୍', 'େଳେ', 'ାଇଛି', '▁', 'ା', 'ି', 'ର', '୍', 'କ', 'େ', 'ନ', 'ବ', 'ତ', 'ସ', 'ୁ', 'ପ', 'ମ', 'ହ', 'ଲ', 'ୟ', 'ଦ', 'ୋ', 'ଇ', 'ଥ', 'ଜ', 'ଗ', 'ଟ', 'ୀ', 'ଣ', 'ଏ', 'ଶ', 'ଙ', 'ଛ', 'ଯ', 'ଆ', 'ଳ', 'ଭ', 'ଷ', 'ଅ', 'ଚ', 'ଧ', 'ଡ', 'ଉ', 'ଖ', 'ଁ', 'ଂ', 'ଓ', 'ୂ', '଼', 'ଠ', 'ୱ', 'ଫ', 'ୃ', '୧', 'ଘ', '୨', '୦', 'ଞ', '୩', '୫', '୪', '୯', 'ଢ', '୬', 'ୌ', '୭', '୮', 'ୈ', 'ଝ', 'ଵ', 'ଃ', 'ଋ', 'ଔ', 'ଈ', 'ଐ', 'ଊ', '\u0b64', 'ୢ', 'ୖ', 'ୠ', 'ୄ', '\u0b04', 'ୗ', '୰', 'ଽ', 'ଌ', '\u0b65', '୲', '୳', '\u0b0d', '\u0b49']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="or", base_path=base_path)
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
            vocab_start_index = 3584
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
