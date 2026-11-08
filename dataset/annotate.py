import glob
import os

import cv2
import numpy as np

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
SCALE = 0.16

os.makedirs(MASK_DIR, exist_ok=True)
points = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))


def draw_preview(small):
    view = small.copy()
    for p in points:
        cv2.circle(view, p, 4, (0, 0, 255), -1)
    if len(points) == 4:
        overlay = view.copy()
        cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0))
        view = cv2.addWeighted(overlay, 0.4, view, 0.6, 0)
    return view


cv2.namedWindow("annotate")
cv2.setMouseCallback("annotate", on_click)

for path in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))):
    name = os.path.splitext(os.path.basename(path))[0]
    mask_path = os.path.join(MASK_DIR, name + ".png")
    if os.path.exists(mask_path):
        continue
    image = cv2.imread(path)
    small = cv2.resize(image, None, fx=SCALE, fy=SCALE)
    points.clear()
    while True:
        cv2.imshow("annotate", draw_preview(small))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key == ord("q"):
            raise SystemExit
        elif key == 13 and len(points) == 4:
            corners = (np.array(points) / SCALE).astype(np.int32)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [corners], 255)
            cv2.imwrite(mask_path, mask)
            break

cv2.destroyAllWindows()