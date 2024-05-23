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
        self.vocab = ['<unk>', '▁ప', 'ని', 'ార', '▁క', '్ర', '▁వ', 'న్', '▁అ', '▁స', '▁మ', 'ంద', 'లు', 'ర్', '▁చ', 'లో', 'స్', '▁త', 'కు', '్య', 'న్న', 'ాల', 'ంచ', 'ారు', 'గా', '▁ర', 'ను', 'క్', 'ంది', 'తు', 'ట్', '▁ప్ర', '▁ఆ', 'రి', '▁ఉ', 'డు', 'ల్', '▁ద', 'ంట', '▁ఇ', '▁చే', '▁న', 'ిన', 'ంత', 'కి', 'ాయ', 'ించ', '▁బ', 'రు', '▁వి', 'డి', 'ప్', 'త్', '▁ఎ', '▁గ', '▁జ', 'ంలో', 'లి', 'టి', 'తి', 'ప్ప', 'డా', 'ంగా', 'లా', 'మా', '▁ఈ', 'సు', 'ార్', '▁ని', 'టు', 'స్తు', 'లే', 'స్త', 'ద్', 'ాయి', 'చ్', 'ది', 'నే', '▁కా', 'దు', '▁రా', 'న్ని', 'న్నారు', 'ష్', 'ాలు', 'చ్చ', 'ండ', 'గు', 'వా', '▁తె', 'సి', '▁భ', 'ిత', '▁స్', 'ందు', 'టీ', 'ంగ', '్యా', 'తో', 'వు', 'డ్', '▁హ', 'నికి', 'ిక', 'పు', '▁శ', '▁చె', 'ళ్', 'యా', 'రో', '▁ఏ', '▁మీ', 'ైన', '▁ఒ', '▁కూ', 'ామ', 'క్క', '▁కొ', '▁సి', '▁మా', 'క్ష', 'ారి', '▁పె', 'ేశ', '▁ఉన్న', 'ాల్', 'పో', 'ంతో', '▁ము', 'మ్', '▁లే', 'ంచి', '▁కూడా', '▁వె', 'కో', 'వి', 'త్ర', 'ట్ట', '▁సం', '▁తీ', '▁కో', 'పై', '▁పో', 'ాడు', '▁కే', '▁ఫ', 'ష్ట', 'డం', 'ప్పు', 'రా', '▁అయ', 'లకు', '▁ను', 'ింది', 'ారం', 'లను', 'కా', 'ద్ద', '▁ఒక', '▁లో', 'మి', 'లీ', 'నా', 'యం', '్వ', '▁పా', 'మె', 'నీ', '▁కు', '▁సమ', 'త్త', 'యు', 'పీ', '▁', '్', 'ి', 'ా', 'ు', 'ర', 'న', 'ం', 'ల', 'క', 'త', 'ప', 'స', 'వ', 'ద', 'మ', 'ే', 'ో', 'య', 'ట', 'చ', 'డ', 'గ', 'ీ', 'ె', 'అ', 'జ', 'బ', 'ూ', 'శ', 'ై', 'ష', 'ఆ', 'ధ', 'ఉ', 'హ', 'భ', 'ొ', 'ఇ', 'ణ', 'ఎ', 'ళ', 'ఈ', 'థ', 'ఫ', 'ఖ', 'ఏ', 'ఒ', 'ృ', 'ౌ', 'ఓ', 'ఘ', 'ఐ', 'ఠ', 'ఛ', 'ఊ', 'ఢ', 'ఞ', 'ఔ', 'ః', '౦', 'ఝ', 'ఋ', 'ఁ', 'ఱ', 'ఙ', 'ౄ', '౩', '౧', 'ఽ', '౨', '౹', 'ౖ', '౯', '౫', '౮', 'ౠ', '౬', '౭', 'ఌ', 'ౙ', '౪', '\u0c65']
        # self.vocab += ["<s>"]
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="te", base_path=base_path)
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
            vocab_start_index = 5120
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
                # transcript = np.array([transcript])
            out_numpy = np.array(transcripts).astype(self.output0_dtype)

            out_tensor_0 = pb_utils.Tensor("TRANSCRIPT", out_numpy)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses
