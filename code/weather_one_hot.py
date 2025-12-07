import numpy as np

def convert_weather_to_one_hot(weather_logits):
    """
    Input: 5-dim cosine similarity vector
           [clear, fog, rain, night, snow]

    Output: 5-dim one-hot weather vector.
    """
    idx = np.argmax(weather_logits)
    one_hot = np.zeros(5, dtype=np.float32)
    one_hot[idx] = 1.0
    return one_hot
