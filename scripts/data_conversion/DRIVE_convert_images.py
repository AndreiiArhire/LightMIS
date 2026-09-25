# https://www.kaggle.com/datasets/zionfuo/drive2004

from PIL import Image
import glob
import os

input_dirs = [
    "/path/to/DRIVE/test/images",
    "/path/to/DRIVE/training/images",
]
out_dir = "/path/to/DRIVE/DRIVE/imagesTr"

os.makedirs(out_dir, exist_ok=True)

image_paths = []
for input_dir in input_dirs:
    image_paths.extend(sorted(glob.glob(os.path.join(input_dir, "*.tif"))))

for i, path in enumerate(image_paths):
    case_id = f"case{i:03d}"
    out_path = os.path.join(out_dir, f"{case_id}_0000.png")

    with Image.open(path) as img:
        img.save(out_path)

print(f"Done")
