from pathlib import Path

import cv2

IMAGE_DIR = Path(__file__).parent / "images"
PATTERN_SIZE = (9, 7)  # inner corners: 9 across, 7 down
EXPECTED_SIZE = (1836, 4080)  # width, height

paths = sorted(IMAGE_DIR.glob("*.jpg"))
found, failed = [], []

for path in paths:
    img = cv2.imread(str(path))
    height, width = img.shape[:2]
    if (width, height) != EXPECTED_SIZE:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, _ = cv2.findChessboardCornersSB(
        gray, PATTERN_SIZE, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    (found if ok else failed).append(path.name)

print(f"Corners found in {len(found)} of {len(found) + len(failed)} portrait images")
for name in failed:
    print(f"FAILED: {name}")