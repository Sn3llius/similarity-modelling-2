# %%
import numpy as np
import librosa
from pathlib import Path
import pickle

# %%
def load_samples():
    data_path =  Path('/home/jakob/Local/similarity-modelling-2/cache/audio')

    samples = []

    for ii, path in enumerate(data_path.iterdir()):
        if ii > 5:
            break

        sample = pickle.load(path.open('rb'))

        # Drop one channel
        sample = sample[None, 0, :]

        samples.append(sample)

    return np.concatenate(samples, axis=0)

samples = load_samples()
print(samples.shape)

# %% MFCC
mfccs = librosa.feature.mfcc(
    samples[0, :],
    sr=48000
)
print(mfccs.shape)



# %%
