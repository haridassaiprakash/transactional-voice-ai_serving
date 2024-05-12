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
        self.vocab = ['<unk>', '▁କ', '▁ସ', '▁ପ', 'ାର', '▁ବ', '୍ର', '୍ତ', 'ରେ', '୍ୟ', '▁ମ', 'ଙ୍', 'ିବ', '▁ଏ', 'ଙ୍କ', '▁କର', 'ାନ', '▁ନ', '▁ଅ', '▁ଦ', '▁ହ', 'ାଇ', 'ିଲ', '▁ଆ', 'ର୍', 'ନ୍ତ', 'ହି', '▁ର', 'ିକ', 'ିବା', '▁ଜ', 'ଛି', 'କୁ', '▁ପ୍ର', '▁ତ', 'ଥିଲ', '▁ଗ', '▁ବି', '▁ଭ', 'ନ୍', '▁ଯ', 'ିତ', 'ନ୍ତି', '▁କରି', '▁ସେ', 'ରୁ', '▁ଲ', 'େବ', '୍ଷ', '▁ଓ', 'ମ୍', '▁ଶ', 'ନା', '▁ଚ', 'ଛନ୍ତି', 'ଟି', 'ହା', 'ୋଇ', '▁ଉ', 'ାରେ', '▁ହୋଇ', 'ଡ଼', 'ଥିବା', '▁ନି', 'ଲି', 'ଷ୍', 'ାମ', 'ଥିଲେ', 'େଇ', 'ଣ୍', 'ତି', '୍ୱ', '୍ଥ', 'ରି', 'ଙ୍କୁ', 'ାସ', 'କ୍ଷ', 'ାବ', 'ମାନ', 'ଧ୍ୟ', '▁ପାଇ', 'ନ୍ଦ', '▁ଏହି', 'ାତ', '▁ରା', '୍ୟା', '▁ସମ', 'ସି', 'ାଯ', '▁ଅନ', 'ଥିଲା', '▁ତା', 'େଶ', 'ୋଗ', '୍ଚ', 'ବା', '▁୧', '▁ମଧ୍ୟ', '▁ଖ', '▁ଘ', '▁ପାଇଁ', 'ିନ', 'ଡି', '୍ରୀ', '▁କି', 'ାଗ', '▁ଟ', '୍ଲ', 'ୁର', 'ଦ୍', 'ାପ', '▁ଜଣ', 'କାର', '▁୨', '▁ଏବ', 'ାଳ', 'ାୟ', 'କ୍ତ', '▁ଫ', '▁କରାଯ', 'ାର୍', 'ଙ୍ଗ', 'ୋଲି', 'େଳ', 'ସ୍ତ', '▁ସଂ', 'ାରୁ', '▁ହେ', 'ଷ୍ଟ', '▁ରାଜ', 'ଯ୍ୟ', 'ୋକ', '▁ଏହା', '▁ମୁ', 'ତା', '▁ଏକ', 'ାଲ', 'ଭି', 'ଳି', 'ାଣ', '▁ମି', '▁ସହ', '▁ସୁ', '▁ହେବ', '▁ପର', '▁କରିବା', 'ାଉ', 'ଡ଼ି', 'ାରି', '▁କେ', '▁ଉପ', 'ଣ୍ଡ', 'ଥା', 'ସ୍ଥ', 'କ୍ର', 'ାନ୍ତ', 'ଦ୍ଧ', 'ଲେ', 'ୀୟ', 'ଞ୍ଚ', '▁ଏବଂ', '▁ଯେ', 'ୃତ', '▁ଧ', '▁ପରେ', '▁ସମ୍', 'େଳେ', 'ାଇଛି', '▁', 'ା', 'ି', 'ର', '୍', 'କ', 'େ', 'ନ', 'ବ', 'ତ', 'ସ', 'ୁ', 'ପ', 'ମ', 'ହ', 'ଲ', 'ୟ', 'ଦ', 'ୋ', 'ଇ', 'ଥ', 'ଜ', 'ଗ', 'ଟ', 'ୀ', 'ଣ', 'ଏ', 'ଶ', 'ଙ', 'ଛ', 'ଯ', 'ଆ', 'ଳ', 'ଭ', 'ଷ', 'ଅ', 'ଚ', 'ଧ', 'ଡ', 'ଉ', 'ଖ', 'ଁ', 'ଂ', 'ଓ', 'ୂ', '଼', 'ଠ', 'ୱ', 'ଫ', 'ୃ', '୧', 'ଘ', '୨', '୦', 'ଞ', '୩', '୫', '୪', '୯', 'ଢ', '୬', 'ୌ', '୭', '୮', 'ୈ', 'ଝ', 'ଵ', 'ଃ', 'ଋ', 'ଔ', 'ଈ', 'ଐ', 'ଊ', '\u0b64', 'ୢ', 'ୖ', 'ୠ', 'ୄ', '\u0b04', 'ୗ', '୰', 'ଽ', 'ଌ', '\u0b65', '୲', '୳', '\u0b0d', '\u0b49']
        
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
