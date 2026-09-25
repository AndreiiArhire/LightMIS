# KvasirSEG: https://datasets.simula.no/kvasir-seg/

from PIL import Image
import numpy as np
import glob, os

in_dir = "/path/to/Kvasir-SEG/masks"
out_dir = "/path/to/Kvasir-SEG/labelsTr"
os.makedirs(out_dir, exist_ok=True)

for i, p in enumerate(sorted(glob.glob(os.path.join(in_dir, "*.jpg")))):
    img = Image.open(p).convert("L")

    img = img.resize((256, 256), Image.NEAREST)

    arr = np.array(img)

    bin_arr = (arr > 127).astype(np.uint8)

    case_id = f"case{i:03d}"
    out_path = os.path.join(out_dir, case_id + ".png")
    Image.fromarray(bin_arr, mode="L").save(out_path)
print(f"Done")
