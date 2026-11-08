from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = Path(__file__).parent / "images"
PATTERN_SIZE = (9, 7)
IMAGE_SIZE = (1836, 4080)

print("name | board_area_% | centre_x_% | centre_y_% | sharpness | width_ratio | height_ratio")

for path in sorted(IMAGE_DIR.glob("*.jpg")):
    img = cv2.imread(str(path))
    height, width = img.shape[:2]
    if (width, height) != IMAGE_SIZE:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(
        gray, PATTERN_SIZE, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if not ok:
        print(f"{path.name} | corners NOT found")
        continue
    pts = corners.reshape(PATTERN_SIZE[1], PATTERN_SIZE[0], 2)
    hull_area = cv2.contourArea(cv2.convexHull(pts.reshape(-1, 2).astype(np.float32)))
    area_pct = 100 * hull_area / (width * height)
    cx, cy = pts.reshape(-1, 2).mean(axis=0)
    x0, y0 = pts.reshape(-1, 2).min(axis=0).astype(int)
    x1, y1 = pts.reshape(-1, 2).max(axis=0).astype(int)
    sharp = cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var()
    top = np.linalg.norm(pts[0, -1] - pts[0, 0])
    bottom = np.linalg.norm(pts[-1, -1] - pts[-1, 0])
    left = np.linalg.norm(pts[-1, 0] - pts[0, 0])
    right = np.linalg.norm(pts[-1, -1] - pts[0, -1])
    print(
        f"{path.name} | {area_pct:.1f} | {100 * cx / width:.0f} | "
        f"{100 * cy / height:.0f} | {sharp:.0f} | "
        f"{top / bottom:.2f} | {left / right:.2f}"
    )