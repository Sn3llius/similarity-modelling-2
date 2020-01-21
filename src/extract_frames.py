"""
This script splits the video files into individual frames and stores them as
JPEG images. The script doesn't perform any classification or other ML task. The
frames are stored in the cache directory and can then be manually split into
classes.
"""

# %%
import av
from pathlib import Path

# %%
root_path = Path().parent
data_path = root_path / 'data'
cache_path = root_path / 'cache'

cache_path.mkdir(parents=False, exist_ok=True)

# %% Iterator over video frames
def iter_frames(path: Path, only_keyframes: bool):
    container = av.open(path.open('rb'))
    stream = container.streams.video[0]

    if only_keyframes:
        stream.codec_context.skip_frame = 'NONKEY'

    for frame in container.decode(stream):
        yield frame.to_image()

#  %% Extract and save the frames
for vid_ii, vid_path in enumerate((data_path / 'videos').iterdir(), start=1):
    # Extract the video index from the file name to make sure they match
    for frame_ii, frame in enumerate(iter_frames(vid_path, only_keyframes=False), start=1):

        # Only extract every nth frame
        if frame_ii % 50 != 0:
            continue

        # Don't save frames that already exists
        opath = cache_path / 'frames' / f'{vid_path.stem}_f{frame_ii}.jpg'
        if opath.exists():
            continue

        print(f'{vid_path.name}  / Frame #{frame_ii}')
        frame.save(opath)

# %%
