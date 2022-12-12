import numpy as np

def load_sample(sound: str, duration: int):
    k = np.load(sound)
    # Find the position of the highest absolute value of the signal
    index = np.argmax(k)
    # start of the signal
    return k[index: index + duration]
# what does "offset" really mean?
# why we need that in our implementation?


# def compute_frequency(signal):
