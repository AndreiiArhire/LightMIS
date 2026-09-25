# https://www.kaggle.com/datasets/zionfuo/drive2004

from PIL import Image
import numpy as np
import glob
import os

in_dirs = [
    "/path/to/DRIVE/test/1st_manual",
    "/path/to/DRIVE/training/1st_manual",
]
out_dir = "/path/to/DRIVE/DRIVE/labelsTr"
os.makedirs(out_dir, exist_ok=True)

gif_paths = [
    p
    for in_dir in in_dirs
    for p in sorted(glob.glob(os.path.join(in_dir, "*.gif")))
]

for i, p in enumerate(gif_paths):
    arr = np.array(Image.open(p).convert("L"))
    bin_arr = (arr > 127).astype(np.uint8)

    out_path = os.path.join(out_dir, f"case{i:03d}.png")
    Image.fromarray(bin_arr, mode="L").save(out_path)

print(f"Done")
