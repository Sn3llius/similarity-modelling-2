"""
Copies the class labels from visual frames to MFCC features.

This script expects that
  - the video frames have been extracted and classified. One class per directory
  - the MFCC features have been extracted and pickled

For each instance the script locates the visual frame's corresponding JPEG
image. Depending on the directory it is in ("kermit" / "non_kermit") the correct
class is assigned.

Results are pickled and stored in the cache directory.
"""

# %%
from pathlib import Path
import numpy as np
import pickle
import gc

# %%
# root_path = Path().parent
root_path = Path('/home/jakob/Local/similarity-modelling-2')
data_path = root_path / 'data'
cache_path = root_path / 'cache'

cache_path.mkdir(parents=False, exist_ok=True)

# %% Label the data
all_mfccs = []
all_labels = []

for mfcc_path in (cache_path/ 'audio').glob('*_mfcc.pickle'):
    print(f'Labelling {mfcc_path.name}')
    cur_mfccs = pickle.load(mfcc_path.open('rb'))
    cur_mfccs = cur_mfccs.transpose()

    # Check if the frame is classified as kermit or non-kermit
    base_name = mfcc_path.name[:-len('_mfcc.pickle')]
    frame_name = f'{base_name}_f{{}}.jpg'
    kermit_path = data_path / 'frames' / 'kermit'
    non_kermit_path = data_path / 'frames' / 'no_kermit'

    cur_labels = np.empty(
        len(cur_mfccs),
        dtype=bool,
    )

    for ii in range(len(cur_labels)):
        fi = (ii + 1) * 50

        p1 = kermit_path / frame_name.format(fi)
        p2 = non_kermit_path / frame_name.format(fi)

        if p1.exists():
            label = True
        else:
            assert p2.exists(), (ii, fi, cur_mfccs.shape, p1, p2)
            label = False

        cur_labels[ii] = label

    all_mfccs.append(cur_mfccs)
    all_labels.append(cur_labels)

# %%
df = {
    'mfcc': np.concatenate(all_mfccs, axis=0),
    'label': np.concatenate(all_labels, axis=0),
}

pickle.dump(
    df,
    (cache_path / 'audio' / 'audio_features.pickle').open('wb')
)


# %%
