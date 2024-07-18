# models/int64_to_int32_converter/1/model.py

import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the model. This function is called only once when the model is
        loaded. 
        """
        self.input_name = "input"
        self.output_name = "output"

    def execute(self, requests):
        """
        This function is called when an inference is requested for this model. 
        The function gets a list of pb_utils.InferenceRequest as the input.
        """
        responses = []

        for request in requests:
            # Get the input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, self.input_name)
            
            # Convert input tensor to numpy array
            input_array = input_tensor.as_numpy()

            # Perform the conversion from int64 to int32
            output_array = input_array.astype(np.int32)

            # Create the output tensor
            output_tensor = pb_utils.Tensor(self.output_name, output_array)

            # Create InferenceResponse
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])

            # Append the response
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        """
        Finalize the model. This function is called only once when the model is
        unloaded.
        """
        print('Cleaning up...')
