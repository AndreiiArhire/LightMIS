# KvasirSEG: https://datasets.simula.no/kvasir-seg/

from PIL import Image
import glob, os

in_dir = "/path/to/Kvasir-SEG/images"
out_dir = "/path/to/Kvasir-SEG/imagesTr"

os.makedirs(out_dir, exist_ok=True)

for i, p in enumerate(sorted(glob.glob(os.path.join(in_dir, "*.jpg")))):
    img = Image.open(p).convert("RGB")
    img = img.resize((256, 256), Image.LANCZOS)

    case_id = f"case{i:03d}"
    out_path = os.path.join(out_dir, case_id + "_0000.png")
    img.save(out_path)

print(f"Done")
