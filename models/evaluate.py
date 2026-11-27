import argparse
import json
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

TEST_NUMBERS = range(91, 101)
THRESHOLDS = np.arange(0.5, 1.0, 0.05)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(weights, device):
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


def predict_probability(model, image_bgr, device):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1))[None].to(device)
    with torch.no_grad():
        return torch.sigmoid(model(tensor))[0, 0].cpu().numpy()


def largest_component(binary):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
    if count < 2:
        return None
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def average_precision(confidences, is_true_positive, ground_truth_count):
    if not confidences:
        return 0.0
    order = np.argsort(-np.array(confidences))
    tp = np.array(is_true_positive, dtype=float)[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = np.concatenate(([0.0], cum_tp / ground_truth_count, [1.0]))
    precision = np.concatenate(([1.0], cum_tp / (cum_tp + cum_fp), [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--data", default="dataset/kaggle_upload")
    parser.add_argument("--out", default="models/test_metrics.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)

    ious, confidences = [], []
    for number in TEST_NUMBERS:
        name = f"book_{number:03d}"
        image = cv2.imread(os.path.join(args.data, "images", name + ".jpg"))
        truth = cv2.imread(os.path.join(args.data, "masks", name + ".png"), 0) > 127
        probability = predict_probability(model, image, device)
        component = largest_component(probability > 0.5)
        if component is None:
            ious.append(0.0)
            confidences.append(None)
        else:
            ious.append(mask_iou(component, truth))
            confidences.append(float(probability[component].mean()))
        print(f"{name} | IoU {ious[-1]:.4f} | confidence {confidences[-1]}")

    predicted = [i for i, c in enumerate(confidences) if c is not None]
    average_precisions = []
    for threshold in THRESHOLDS:
        hits = [ious[i] >= threshold for i in predicted]
        average_precisions.append(average_precision([confidences[i] for i in predicted], hits, len(ious)))

    true_positives = sum(ious[i] >= 0.5 for i in predicted)
    results = {
        "test_images": len(ious),
        "mean_mask_iou": float(np.mean(ious)),
        "mAP@0.5": average_precisions[0],
        "mAP@0.5:0.95": float(np.mean(average_precisions)),
        "precision@0.5": true_positives / len(predicted) if predicted else 0.0,
        "recall@0.5": true_positives / len(ious),
        "per_image_iou": ious,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "per_image_iou"}, indent=2))


if __name__ == "__main__":
    main()