import math
import io
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

def batchify(arr, batch_size=1):
    try:
        num_batches = math.ceil(len(arr) / batch_size)
        return [arr[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    except Exception as e:
        logger.error(f"Error in batchify: {e}")
        raise

def pad_batch(batch_data):
    try:
        batch_data_lens = np.asarray([len(data) for data in batch_data], dtype=np.int32)
        max_length = max(batch_data_lens)
        batch_size = len(batch_data)

        padded_zero_array = np.zeros((batch_size, max_length), dtype=np.float32)

        for idx, data in enumerate(batch_data):
            padded_zero_array[idx, 0:batch_data_lens[idx]] = data

        return padded_zero_array, np.reshape(batch_data_lens, [-1, 1])
    except Exception as e:
        logger.error(f"Error in pad_batch: {e}")
        raise

def get_raw_audio_from_file_bytes(file_bytes, standard_sampling_rate):
    try:
        file_handle = io.BytesIO(file_bytes)
        raw_audio, _ = librosa.load(file_handle, sr=standard_sampling_rate)
        return raw_audio.astype("float32")
    except Exception as e:
        logger.error(f"Error in get_raw_audio_from_file_bytes: {e}")
        raise
