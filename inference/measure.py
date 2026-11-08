import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "calibration" / "camera_params.npz"
WEIGHTS_PATH = ROOT / "models" / "best.pt"
OUTPUT_DIR = ROOT / "inference" / "outputs"
PATTERN_SIZE = (9, 7)
SQUARE_MM = 20.0  # design value, printed size NOT verified
MODEL_SIZE = (512, 1152)  # width, height the model was trained on
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_AREA_FRACTION = 0.01
MIN_CONFIDENCE = 0.9  # placeholder, not tuned or evaluated


def load_model():
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                state = checkpoint[key]
                break
    model.load_state_dict(state)
    model.eval()
    return model


def segment_book(model, image_bgr):
    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(image_bgr, MODEL_SIZE), cv2.COLOR_BGR2RGB)
    scaled = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(scaled.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].numpy()
    probability = cv2.resize(probability, (width, height))
    mask = (probability > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None, 0.0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    book = (labels == largest).astype(np.uint8)
    if book.sum() < MIN_AREA_FRACTION * width * height:
        return None, 0.0
    return book, float(probability[book > 0].mean())


def order_corners(points):
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return points[np.argsort(angles)]


def book_corners(book_mask):
    contours, _ = cv2.findContours(
        book_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    for epsilon in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2).astype(np.float32))
    box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    return order_corners(box)


def board_homography(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, PATTERN_SIZE, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if not found:
        return None, None
    corners = corners.reshape(-1, 2).astype(np.float32)
    board_mm = (
        np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(-1, 2)
        * SQUARE_MM
    ).astype(np.float32)
    homography, _ = cv2.findHomography(corners, board_mm)
    return homography, corners


def measure_mm(corners_px, homography):
    mapped = cv2.perspectiveTransform(
        corners_px.reshape(-1, 1, 2), homography
    ).reshape(4, 2)
    sides = [np.linalg.norm(mapped[i] - mapped[(i + 1) % 4]) for i in range(4)]
    first = (sides[0] + sides[2]) / 2
    second = (sides[1] + sides[3]) / 2
    return max(first, second), min(first, second)


def save_overlay(image_bgr, corners_px, text, out_path):
    overlay = image_bgr.copy()
    cv2.polylines(overlay, [corners_px.astype(np.int32)], True, (0, 255, 0), 8)
    top_left = corners_px.min(axis=0).astype(int)
    cv2.putText(
        overlay, text, (int(top_left[0]), max(int(top_left[1]) - 30, 80)),
        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6,
    )
    small = cv2.resize(overlay, None, fx=0.5, fy=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), small)


def main():
    parser = argparse.ArgumentParser(description="Measure a book cover in mm.")
    parser.add_argument("image")
    parser.add_argument("--no-undistort", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: could not read image {args.image}")
        return 1

    model = load_model()
    start = time.perf_counter()

    if not args.no_undistort:
        params = np.load(PARAMS_PATH)
        image = cv2.undistort(image, params["camera_matrix"], params["dist_coeffs"])

    book_mask, confidence = segment_book(model, image)
    if book_mask is None:
        print("BOOK NOT FOUND: no measurement reported.")
        return 1

    homography, _ = board_homography(image)
    if homography is None:
        print("REFERENCE NOT FOUND (checkerboard): no measurement reported.")
        return 1

    if confidence < MIN_CONFIDENCE:
        print(f"LOW CONFIDENCE ({confidence:.3f}): refusing to measure.")
        return 1

    corners = book_corners(book_mask)
    height_mm, width_mm = measure_mm(corners, homography)
    elapsed = time.perf_counter() - start

    name = Path(args.image).stem + ("_raw" if args.no_undistort else "")
    text = f"W {width_mm:.1f} mm  H {height_mm:.1f} mm  conf {confidence:.3f}"
    save_overlay(image, corners, text, OUTPUT_DIR / f"{name}_overlay.jpg")

    print(f"width_mm={width_mm:.2f} height_mm={height_mm:.2f} confidence={confidence:.4f}")
    print(f"undistorted={not args.no_undistort} seconds={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())