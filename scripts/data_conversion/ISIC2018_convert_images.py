# https://challenge.isic-archive.com/data/#2018

from pathlib import Path
from PIL import Image

in_dirs = [
    Path("/path/to/isic2018/ISIC2018_Task1-2_Test_Input"),
    Path("/path/to/isic2018/ISIC2018_Task1-2_Validation_Input"),
    Path("/path/to/isic2018/ISIC2018_Task1-2_Training_Input"),
]
out_dir = Path("/path/to/isic2018/imagesTr")
out_dir.mkdir(parents=True, exist_ok=True)

jpg_files = sorted(
    (
        path
        for folder in in_dirs
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    ),
    key=lambda path: path.name,
)

for i, path in enumerate(jpg_files):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img.save(out_dir / f"case{i:03d}_0000.png")

print(f"Done")
