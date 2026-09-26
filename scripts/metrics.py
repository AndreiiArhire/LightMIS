import os
import numpy as np
import cv2

from medpy.metric.binary import dc, jc, hd95, assd


base_dir = "/path/to/nnUNet_results/Dataset100_KvasirSEG/nnUNetTrainer_LightMIS__nnUNetPlans__2d"
label_dir = "/path/to/nnUNet_raw/Dataset100_KvasirSEG/labelsTr"

folds = [0, 1, 2, 3, 4]


def load_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to load {path}")

    return (img > 0).astype(np.uint8)


def compute_metrics(pred, gt):
    dice = dc(pred, gt)
    iou = jc(pred, gt)

    if pred.sum() == 0 and gt.sum() == 0:
        return dice, iou, 0.0, 0.0

    if pred.sum() == 0 or gt.sum() == 0:
        return dice, iou, np.nan, np.nan

    try:
        hd = hd95(pred, gt)
        asd = assd(pred, gt)
    except Exception:
        hd = np.nan
        asd = np.nan

    return dice, iou, asd, hd


def evaluate_fold(fold_id, log_lines):
    input_dir = os.path.join(base_dir, f"fold_{fold_id}/validation")

    results = []

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.startswith("case") and f.endswith(".png")
    ])

    for fname in files:
        pred_path = os.path.join(input_dir, fname)
        gt_path = os.path.join(label_dir, fname)

        if not os.path.exists(gt_path):
            continue

        pred = load_mask(pred_path)
        gt = load_mask(gt_path)

        dice, iou, asd, hd = compute_metrics(pred, gt)
        results.append((dice, iou, asd, hd))

    results = np.array(results, dtype=np.float32)

    if len(results) == 0:
        raise ValueError(f"No valid results found for fold {fold_id}")

    dice_vals = results[:, 0]
    iou_vals = results[:, 1]
    asd_vals = results[:, 2]
    hd_vals = results[:, 3]

    asd_valid = asd_vals[~np.isnan(asd_vals)]
    hd_valid = hd_vals[~np.isnan(hd_vals)]

    dice_mean = np.mean(dice_vals)
    iou_mean = np.mean(iou_vals)
    asd_mean = np.mean(asd_valid)
    hd_mean = np.mean(hd_valid)

    line = f"\n===== FOLD {fold_id} ====="
    print(line)
    log_lines.append(line)

    line = f"Dice → {dice_mean * 100:.2f} ± {np.std(dice_vals) * 100:.2f}"
    print(line)
    log_lines.append(line)

    line = f"IoU  → {iou_mean * 100:.2f} ± {np.std(iou_vals) * 100:.2f}"
    print(line)
    log_lines.append(line)

    line = f"ASD  → {asd_mean:.2f} ± {np.std(asd_valid):.2f}"
    print(line)
    log_lines.append(line)

    line = f"HD95 → {hd_mean:.2f} ± {np.std(hd_valid):.2f}"
    print(line)
    log_lines.append(line)

    return (
        dice_mean,
        iou_mean,
        asd_mean,
        hd_mean
    )


def evaluate():
    log_lines = []

    dice_folds = []
    iou_folds = []

    asd_folds = []
    hd_folds = []

    for f in folds:
        dice_mean, iou_mean, asd_mean, hd_mean = evaluate_fold(f, log_lines)

        dice_folds.append(dice_mean)
        iou_folds.append(iou_mean)

        asd_folds.append(asd_mean)
        hd_folds.append(hd_mean)

    dice_folds = np.array(dice_folds, dtype=np.float32)
    iou_folds = np.array(iou_folds, dtype=np.float32)

    asd_folds = np.array(asd_folds, dtype=np.float32)
    hd_folds = np.array(hd_folds, dtype=np.float32)

    header = "\n==============================\n===== FINAL (OVER FOLDS) =====\n=============================="
    print(header)
    log_lines.append(header)

    line = f"Dice → {np.mean(dice_folds) * 100:.2f} ± {np.std(dice_folds) * 100:.2f}"
    print(line)
    log_lines.append(line)

    line = f"IoU  → {np.mean(iou_folds) * 100:.2f} ± {np.std(iou_folds) * 100:.2f}"
    print(line)
    log_lines.append(line)

    line = f"HD95 → {np.mean(hd_folds):.2f} ± {np.std(hd_folds):.2f}"
    print(line)
    log_lines.append(line)

    line = f"ASD  → {np.mean(asd_folds):.2f} ± {np.std(asd_folds):.2f}"
    print(line)
    log_lines.append(line)

    output_file = os.path.join(base_dir, "evaluation_results.txt")

    with open(output_file, "w") as f:
        for l in log_lines:
            f.write(l + "\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    evaluate()
