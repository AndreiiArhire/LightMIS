# https://challenge.isic-archive.com/data/#2017

from PIL import Image
import numpy as np
import glob
import os

in_dirs = [
    "/path/to/isic2017/ISIC-2017_Test_v2_Part1_GroundTruth",
    "/path/to/isic2017/ISIC-2017_Training_Part1_GroundTruth",
    "/path/to/isic2017/ISIC-2017_Validation_Part1_GroundTruth",
]
out_dir = "/path/to/isic2017/labelsTr"

os.makedirs(out_dir, exist_ok=True)

paths = sorted(
    (
        p
        for in_dir in in_dirs
        for p in glob.glob(os.path.join(in_dir, "*.png"))
    ),
    key=os.path.basename,
)

for i, p in enumerate(paths):
    img = Image.open(p).convert("L")
    img = img.resize((256, 256), Image.NEAREST)

    arr = np.array(img)
    bin_arr = (arr > 127).astype(np.uint8)

    case_id = f"case{i:03d}"
    out_path = os.path.join(out_dir, case_id + ".png")
    Image.fromarray(bin_arr, mode="L").save(out_path)
