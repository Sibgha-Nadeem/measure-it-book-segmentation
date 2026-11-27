import glob
import os
import shutil

import cv2

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
SPLIT_FILE = "dataset/splits/split.txt"
OUT_DIR = "dataset/kaggle_upload"
SIZE = (512, 1152)

shutil.rmtree(OUT_DIR, ignore_errors=True)
os.makedirs(os.path.join(OUT_DIR, "images"))
os.makedirs(os.path.join(OUT_DIR, "masks"))

for path in sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))):
    name = os.path.splitext(os.path.basename(path))[0]
    image = cv2.resize(cv2.imread(path), SIZE, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(cv2.imread(os.path.join(MASK_DIR, name + ".png"), 0), SIZE, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT_DIR, "images", name + ".jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(OUT_DIR, "masks", name + ".png"), mask)

shutil.copy(SPLIT_FILE, OUT_DIR)
shutil.make_archive(OUT_DIR, "zip", OUT_DIR)
print("zip size (MB):", round(os.path.getsize(OUT_DIR + ".zip") / 1e6, 1))