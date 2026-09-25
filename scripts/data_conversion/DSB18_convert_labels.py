# https://www.kaggle.com/competitions/data-science-bowl-2018/data

from pathlib import Path

import numpy as np
from PIL import Image

input_dir = Path("/path/to/data-science-bowl-2018/stage1_train")
out_masks = Path("/path/to/data-science-bowl-2018/masks")
out_labels = Path("/path/to/data-science-bowl-2018/labelsTr")


def main():
    out_masks.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    cases = sorted(
        (p for p in input_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )

    for i, case in enumerate(cases):
        masks = []
        for mask_file in (case / "masks").iterdir():
            with Image.open(mask_file) as mask:
                masks.append(np.array(mask.convert("L")))

        combined = np.maximum.reduce(masks)
        binary = (combined > 0).astype(np.uint8) * 255
        merged = Image.fromarray(binary, mode="L")
        merged.save(out_masks / f"{case.name}.png")

        resized = merged.resize((256, 256), Image.Resampling.NEAREST)
        label = (np.array(resized) > 127).astype(np.uint8)
        Image.fromarray(label, mode="L").save(out_labels / f"case{i:03d}.png")

    print(f"Done")


if __name__ == "__main__":
    main()
