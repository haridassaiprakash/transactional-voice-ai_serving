import numpy as np
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        """
        `initialize` is called only once when the model is being loaded.
        This function allows the model to initialize any state associated with this model.

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
        model_name = 'ai4bharat/indic-bert'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.labels_to_ids = {'balance_check': 0, 'cancel': 1, 'confirm': 2, 'electricity_payment': 3, 'emi_collection_full': 4, 'emi_collection_partial': 5, 'fastag_recharge': 6, 'gas_payment': 7, 'inform': 8, 'insurance_renewal': 9, 'mobile_recharge_postpaid': 10, 'mobile_recharge_prepaid': 11, 'p2p_transfer': 12, 'petrol_payment': 13, 'upi_creation': 14}
        self.ids_to_labels = {intent_id: intent_label for intent_label, intent_id in self.labels_to_ids.items()}
        self.conf_threshold = 0.75
        self.allowed_intents = {'balance_check', 'fastag_recharge', 'p2p_transfer', 'mobile_recharge_prepaid', 'gas_payment'}

    def softmax(self, logits):
        """
        Compute softmax values for each set of scores in logits.

        Parameters
        ----------
        logits : numpy.ndarray
          The logits to compute the softmax over.

        Returns
        -------
        numpy.ndarray
          The softmax probabilities.
        """
        exp_scores = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    def execute(self, requests):
        """
        `execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest.

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`.
        """
        responses = []

        for request in requests:
            # Get the input tensor 'logits' from the request
            logits = pb_utils.get_input_tensor_by_name(request, 'logits').as_numpy()
            probs = self.softmax(logits)
            max_probs = np.max(probs, axis=-1)
            preds = np.argmax(probs, axis=-1)

            batch_predictions = []
            for prob, pred in zip(max_probs, preds):
                intent = self.ids_to_labels[pred]
                if prob < self.conf_threshold or intent not in self.allowed_intents:
                    _prediction = "unknown"
                else:
                    _prediction = intent
                batch_predictions.append(_prediction)

            # Create the output tensor with the predictions
            labels = np.array([l.encode('utf-8') for l in batch_predictions])
            labels_tensor = pb_utils.Tensor("labels", labels)
            # Create an inference response with the output tensor
            inference_response = pb_utils.InferenceResponse(output_tensors=[labels_tensor])
            responses.append(inference_response)

        return responses
