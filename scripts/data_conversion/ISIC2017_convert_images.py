# https://challenge.isic-archive.com/data/#2017

from PIL import Image
import glob
import os

in_dirs = [
    "/path/to/isic2017/ISIC-2017_Test_v2_Data",
    "/path/to/isic2017/ISIC-2017_Training_Data",
    "/path/to/isic2017/ISIC-2017_Validation_Data",
]
out_dir = "/path/to/isic2017/imagesTr"

os.makedirs(out_dir, exist_ok=True)

paths = sorted(
    (
        p
        for in_dir in in_dirs
        for p in glob.glob(os.path.join(in_dir, "*.jpg"))
    ),
    key=os.path.basename,
)

for i, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    img = img.resize((256, 256), Image.LANCZOS)

    case_id = f"case{i:03d}"
    out_path = os.path.join(out_dir, case_id + "_0000.png")
    img.save(out_path)
