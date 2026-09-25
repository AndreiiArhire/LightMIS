# https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset/data

import re
from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path("/path/to/Dataset_BUSI_with_GT")
BENIGN_DIR = ROOT_DIR / "benign"
MALIGNANT_DIR = ROOT_DIR / "malignant"
OUTPUT_DIR = ROOT_DIR / "labelsTr"

MASK_PATTERN = re.compile(r"(.+)_mask(?:_\d+)?$", re.IGNORECASE)


def collect_cases(folders: list[Path]) -> list[tuple[Path, list[Path]]]:
    cases = []
    for folder in folders:
        files = sorted(
            (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
            key=lambda p: p.name,
        )
        images = []
        masks_by_stem: dict[str, list[Path]] = {}

        for file in files:
            match = MASK_PATTERN.fullmatch(file.stem)
            if match:
                masks_by_stem.setdefault(match.group(1), []).append(file)
            else:
                images.append(file)

        image_stems = {image.stem for image in images}
        images_without_masks = sorted(image_stems - masks_by_stem.keys())
        masks_without_images = sorted(masks_by_stem.keys() - image_stems)
        if images_without_masks or masks_without_images:
            raise ValueError(
                f"Unpaired PNGs in {folder}: "
                f"images without masks={images_without_masks}; "
                f"masks without images={masks_without_images}"
            )

        cases.extend((image, masks_by_stem[image.stem]) for image in images)

    if not cases:
        raise ValueError("No PNG image and mask pairs were found.")
    return cases


def main() -> None:
    folders = [BENIGN_DIR, MALIGNANT_DIR]
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Directory not found: {folder}")

    cases = collect_cases(folders)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, (image_path, mask_paths) in enumerate(cases):
        combined = np.zeros((256, 256), dtype=np.uint8)
        for mask_path in mask_paths:
            with Image.open(mask_path) as mask:
                resized = mask.convert("L").resize(
                    (256, 256), resample=Image.Resampling.NEAREST
                )
                binary = (np.asarray(resized) > 127).astype(np.uint8)
            np.maximum(combined, binary, out=combined)

        destination = OUTPUT_DIR / f"case{index:03d}.png"
        Image.fromarray(combined).save(destination)

    print(f"Done")


if __name__ == "__main__":
    main()
