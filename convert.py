"""Convert all .mat and .avifiles to .mp4 in the current directory."""

import subprocess
from scipy import io
from skvideo import io as skio
import os

# get all of the .mat files in the current directory
fns = os.listdir('.')
mat_fns = [fn for fn in fns if fn.endswith('.mat')]
# load each .mat file and convert to .mp4
for mat_fn in mat_fns:
    # load the .mat file
    mat_data = io.loadmat(mat_fn)
    # extract the video data
    video_data = mat_data['vidData'][:, :, 0]
    video_data = video_data.transpose((2, 0, 1))
    # write the video data to a .mp4 file
    new_fn = mat_fn.replace('.mat', '.mp4')
    if not os.path.exists(new_fn):
        skio.vwrite(new_fn, video_data)


# same thing but load .avi files and vwrite to .mp4
avi_fns = [fn for fn in fns if fn.endswith('.avi')]
for avi_fn in avi_fns:
    # run a subprocess command using ffmpeg to convert .avi to .mp4
    output_fn = avi_fn.replace('.avi', '.mp4')
    if not os.path.exists(output_fn):
        subprocess.run(['ffmpeg', '-i', avi_fn, output_fn])