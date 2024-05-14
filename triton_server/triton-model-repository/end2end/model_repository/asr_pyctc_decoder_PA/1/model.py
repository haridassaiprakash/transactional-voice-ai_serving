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
        self.vocab = ['<unk>', '▁ਸ', '▁ਕ', '▁ਦ', '▁ਹ', '▁ਨ', '▁ਵ', 'ਾਂ', '▁ਪ', '▁ਜ', '▁ਮ', 'ਾਰ', '▁ਅ', '▁ਇ', '▁ਤ', '▁ਬ', '▁ਦੇ', '▁ਲ', '▁ਹੈ', '▁ਵਿ', 'ਾਲ', 'ੂੰ', 'ਿਆ', '▁ਰ', '▁ਦੀ', '▁ਨੂੰ', '▁ਕਿ', '▁ਉ', 'ਸ਼', '▁ਗ', 'ਿੰ', '▁ਕਰ', 'ਤੇ', 'ਾਨ', '▁ਆ', 'ੋਂ', '੍ਰ', 'ਆਂ', '▁ਅਤੇ', '▁ਹੋ', 'ੱਚ', 'ਤੀ', 'ਦਾ', '▁ਚ', '▁ਇਸ', '▁ਦਾ', 'ਤਾ', 'ਰੀ', 'ਹੀ', '▁ਨੇ', 'ਿੱ', '੍ਹ', 'ਦੇ', '▁ਵਿੱਚ', '▁ਲਈ', 'ਜ਼', '▁ਕੀ', '▁ਖ', '▁ਸਿੰ', '▁ਨਾਲ', 'ੱਕ', '▁ਸਿੰਘ', 'ਲਾ', '▁ਪ੍ਰ', 'ਹਿ', '▁ਹਨ', 'ਹਾ', '▁ਸ਼', 'ਦੀ', '▁ਤੋਂ', '▁ਭ', '▁ਜਾ', '▁ਵਿਚ', '▁ਵੀ', '▁ਫ', 'ਰਾ', 'ਵਾ', '▁ਤੇ', 'ਣਾ', '੍ਹਾਂ', '▁ਇਹ', 'ੁਰ', 'ੀਆਂ', 'ਨਾ', 'ਕਾਰ', 'ੁੱ', 'ਵੇ', '▁ਸੀ', '▁ਡ', '▁ਸਕ', 'ਦਰ', 'ਟੀ', 'ਨ੍ਹਾਂ', 'ਸੀ', '▁ਕੇ', 'ੱਲ', 'ਹੀਂ', '▁ਸਮ', '▁ਇੱਕ', 'ਿਸ', '▁ਕਰਨ', '▁ਐ', 'ਾਈ', '▁ਨਹੀਂ', 'ੁੰ', 'ਨੀ', 'ਿਆਂ', '▁ਜਾਂ', 'ਕੇ', '▁ਆਪ', '▁ਕੋ', '▁ਸੰ', 'ਿਲ', '▁ਸਾ', 'ੈਂ', '▁ਪਰ', '▁ਉਹ', '▁ਬਾ', 'ਕਾ', '▁ਉਨ੍ਹਾਂ', 'ਲੇ', 'ਲੀ', '▁ਦਿੱ', '▁ਹੀ', 'ਗਾ', '▁ਸਰ', '▁ਰਾ', 'ਣੇ', 'ਜਾ', '▁ਕੀਤਾ', 'ਿਰ', '▁ਉਸ', 'ਰੇ', '▁ਜ਼', '▁ਮੁ', 'ੱਖ', '▁ਲੋ', '▁ਤੁ', '▁ਘ', 'ਤਰ', 'ੰਗ', '▁ਟ', '▁ਗਿਆ', 'ਣੀ', 'ਾਇ', '▁ਕਿਸ', 'ੱਸ', '▁ਕੁ', 'ਿਕ', '▁ਜੋ', 'ੌਰ', '▁ਕਿਹਾ', 'ਜੀ', '▁ਇਕ', 'ਡੀ', '▁ਕੀਤੀ', '▁ਪੰ', '▁ਵਾਲ', 'ਨਾਂ', '▁ਕਾਰ', 'ਾਰੇ', '▁ਯ', 'ਹੇ', '▁ਸੁ', '▁ਤਾਂ', '▁ਪਾ', '▁ਨਾ', 'ਉਣ', 'ਟਰ', '▁ਧ', '▁ਮਾ', 'ਮੀ', 'ਾਰੀ', 'ੂਰ', 'ਹੁ', 'ਫ਼', 'ੰਤ', '▁', 'ਾ', 'ਰ', 'ੀ', 'ਸ', 'ਿ', 'ੇ', 'ਕ', 'ਨ', 'ਦ', 'ਹ', 'ਤ', 'ਲ', 'ਂ', 'ਵ', 'ੰ', 'ਮ', 'ਜ', 'ਪ', 'ੋ', 'ੱ', 'ਆ', 'ੁ', 'ਗ', 'ਬ', '਼', 'ੈ', 'ੂ', 'ਇ', 'ਅ', 'ਚ', 'ਣ', 'ਟ', 'ਈ', 'ਉ', '੍', 'ਖ', 'ਡ', 'ਧ', 'ਫ', 'ਭ', 'ਘ', 'ੜ', 'ੌ', 'ਥ', 'ਏ', 'ਐ', 'ਯ', 'ਛ', 'ਝ', 'ਠ', 'ਓ', 'ਢ', 'ਊ', '੫', 'ਔ', '੧', '੨', '੩', 'ੳ', 'ੲ', '੪', '੦', '੯', '੭', '੬', '੮', 'ਞ', 'ਙ', 'ੴ', 'ਃ', 'ੑ', 'ੵ', 'ਁ', '\u0a5d', '\u0a00', '\u0a0c', '\u0a44', '\u0a7f']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="pa", base_path=base_path)
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
            vocab_start_index = 3840
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
