from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = Path(__file__).parent
IMAGE_DIR = CALIB_DIR / "images"
PATTERN_SIZE = (9, 7)  # inner corners: 9 across, 7 down
IMAGE_SIZE = (1836, 4080)  # width, height

params = np.load(CALIB_DIR / "camera_params.npz")
camera_matrix = params["camera_matrix"]
dist_coeffs = params["dist_coeffs"]


def line_residuals(points):
    centred = points - points.mean(axis=0)
    normal = np.linalg.svd(centred)[2][-1]
    return centred @ normal


def grid_straightness(grid):
    lines = list(grid) + list(grid.transpose(1, 0, 2))
    residuals = np.concatenate([line_residuals(line) for line in lines])
    return np.sqrt(np.mean(residuals**2)), np.abs(residuals).max()


raw_rms, undistorted_rms, raw_max, undistorted_max = [], [], [], []

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
        continue
    raw = corners.reshape(-1, 2).astype(np.float64)
    fixed = cv2.undistortPoints(
        raw.reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=camera_matrix
    ).reshape(-1, 2)
    shape = (PATTERN_SIZE[1], PATTERN_SIZE[0], 2)
    rms, worst = grid_straightness(raw.reshape(shape))
    raw_rms.append(rms)
    raw_max.append(worst)
    rms, worst = grid_straightness(fixed.reshape(shape))
    undistorted_rms.append(rms)
    undistorted_max.append(worst)

improved = sum(u < r for u, r in zip(undistorted_rms, raw_rms))
print(f"Images: {len(raw_rms)}, improved after undistortion: {improved}")
print(f"Mean line-fit RMS (px): raw {np.mean(raw_rms):.3f}, undistorted {np.mean(undistorted_rms):.3f}")
print(f"Mean worst deviation (px): raw {np.mean(raw_max):.3f}, undistorted {np.mean(undistorted_max):.3f}")