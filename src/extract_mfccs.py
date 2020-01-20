"""
Extracts audio features from the videos files.

First extracts the entire audio stream from the video, then uses librosa to
calculate MFCC values. The window size is selected to coincide with the duration
between visual frames.

Results are pickled and written to the cache directory.
"""

# %%
import av
from pathlib import Path
import numpy as np
import pickle
import gc
import pandas as pd
import librosa


# %%
# root_path = Path().parent
root_path = Path('/home/jakob/Local/similarity-modelling-2')
data_path = root_path / 'data'
cache_path = root_path / 'cache'

created_dirs = [
    cache_path,
    cache_path / 'audio'
]
for path in created_dirs:
    path.mkdir(parents=False, exist_ok=True)


# %% Read all audio samples of the file
def get_all_audio(path: Path):
    input_container = av.open(str(path))
    input_stream = input_container.streams.get(audio=0)[0]

    chunks = []
    for ii, frame in enumerate(input_container.decode(input_stream)):
        chunk = frame.to_ndarray()
        chunks.append(chunk)

    return np.concatenate(chunks, axis=1)


# %% Extract and save the MFCCs
fps = 25
sample_rate = 48000
frame_ratio = 50

for vid_path in (data_path / 'videos').glob('*.avi'):
    opath = cache_path / 'audio' / f'{vid_path.stem}_mfcc.pickle'
    print(f'Processing {vid_path.name}')

    if opath.exists():
        print('  Skipping')
        continue

    print('  Collecting audio')
    audio = get_all_audio(vid_path)
    print(audio.shape)

    # Drop one channel
    audio = audio[0, :]

    print('  Calculating MFCC')
    mfccs = librosa.feature.mfcc(
        audio,
        sr=sample_rate,
        hop_length=int(frame_ratio / fps * sample_rate)
    )

    print(f'  MFCC shape: {mfccs.shape}')
    pickle.dump(mfccs, opath.open('wb'))

# %%
