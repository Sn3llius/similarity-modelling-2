# %%
import av
from pathlib import Path
import numpy as np
import pickle
import gc


# %%
root_path = Path().parent.resolve()
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

# %% Extract and save the frames
fps = 25
sample_rate = 44800
frame_ratio = 50
radius = 0.5

for vid_ii, vid_path in enumerate((data_path / 'videos').glob('*.avi'), start=1):
    print(f'Collecting audio for {vid_path.name}')
    audio = get_all_audio(vid_path)

    frame_ii = 0
    while True:
        frame_ii += frame_ratio
        print(f'{vid_path.name}  / Frame #{frame_ii}')

        start = int((frame_ii / fps - radius) * sample_rate)
        end = int(start + 2 * radius * sample_rate)

        if end > audio.shape[1]:
            break

        chunk = audio[:, start:end]
        opath = cache_path / 'audio' / f'{vid_path.stem}_f{frame_ii}.pickle'
        pickle.dump(chunk, opath.open('wb'))

        gc.collect()



# %%

