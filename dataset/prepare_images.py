import csv
import glob
import os
import random

import cv2
import numpy as np

RAW_DIR = "dataset/raw"
OUT_DIR = "dataset/images"
MAP_FILE = "dataset/splits/name_map.csv"
PARAMS = "calibration/camera_params.npz"
SEED = 42

params = np.load(PARAMS)
camera_matrix = params["camera_matrix"]
dist_coeffs = params["dist_coeffs"]

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
assert len(files) == 100, f"expected 100 photos, found {len(files)}"

random.seed(SEED)
train, val, test = [], [], []
for start in range(0, 100, 10):
    group = files[start:start + 10]
    random.shuffle(group)
    train += group[:7]
    val += group[7:9]
    test += group[9:]

os.makedirs(OUT_DIR, exist_ok=True)
with open(MAP_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["new_name", "original_name"])
    for i, path in enumerate(train + val + test, start=1):
        new_name = f"book_{i:03d}.jpg"
        undistorted = cv2.undistort(cv2.imread(path), camera_matrix, dist_coeffs)
        cv2.imwrite(os.path.join(OUT_DIR, new_name), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95])
        writer.writerow([new_name, os.path.basename(path)])

print("done:", len(train), len(val), len(test))