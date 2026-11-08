import sys
from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = Path(__file__).parent
SCALE = 0.3

params = np.load(CALIB_DIR / "camera_params.npz")
camera_matrix = params["camera_matrix"]
dist_coeffs = params["dist_coeffs"]

img = cv2.imread(sys.argv[1])
undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

panels = []
for image in (img, undistorted):
    small = cv2.resize(image, None, fx=SCALE, fy=SCALE)
    for x in range(0, small.shape[1], 100):
        cv2.line(small, (x, 0), (x, small.shape[0]), (0, 0, 255), 1)
    for y in range(0, small.shape[0], 100):
        cv2.line(small, (0, y), (small.shape[1], y), (0, 0, 255), 1)
    panels.append(small)

cv2.imwrite(str(CALIB_DIR / "undistort_demo.jpg"), np.hstack(panels))
print("Saved calibration/undistort_demo.jpg (left: raw, right: undistorted)")