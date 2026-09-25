# https://www.kaggle.com/competitions/data-science-bowl-2018/data

from pathlib import Path

from PIL import Image

input_dir = Path("/path/to/data-science-bowl-2018/stage1_train")
out_images = Path("/path/to/data-science-bowl-2018/images")
out_images_tr = Path("/path/to/data-science-bowl-2018/imagesTr")


def main():
    out_images.mkdir(parents=True, exist_ok=True)
    out_images_tr.mkdir(parents=True, exist_ok=True)

    cases = sorted(
        (p for p in input_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )

    for i, case in enumerate(cases):
        image_file = next((case / "images").iterdir())

        with Image.open(image_file) as image:
            image.save(out_images / f"{case.name}.png")

            resized = image.convert("RGB").resize(
                (256, 256), Image.Resampling.LANCZOS
            )
            resized.save(out_images_tr / f"case{i:03d}_0000.png")

    print(f"Done")


if __name__ == "__main__":
    main()
