from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = Path(__file__).parent
IMAGE_DIR = CALIB_DIR / "images"
PATTERN_SIZE = (9, 7)  # inner corners: 9 across, 7 down
IMAGE_SIZE = (1836, 4080)  # width, height
SQUARE_MM = 20.0  # placeholder until measured with a ruler

object_points_template = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
object_points_template[:, :2] = (
    np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_MM
)

object_points, image_points, used = [], [], []

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
    object_points.append(object_points_template)
    image_points.append(corners)
    used.append(path.name)

rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, IMAGE_SIZE, None, None,
    flags=cv2.CALIB_FIX_K3,
)

print(f"Images used: {len(used)}")
print(f"RMS reprojection error (px): {rms:.4f}")

for name, objp, imgp, rvec, tvec in zip(
    used, object_points, image_points, rvecs, tvecs
):
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(
        imgp.reshape(-1, 2), projected.reshape(-1, 2), cv2.NORM_L2
    ) / np.sqrt(len(projected))
    print(f"{name}: {error:.4f} px")

print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())

np.savez(
    CALIB_DIR / "camera_params.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_size=np.array(IMAGE_SIZE),
    square_mm=SQUARE_MM,
    rms=rms,
)
print("Saved calibration/camera_params.npz")