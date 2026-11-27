from pathlib import Path

import cv2

IMAGE_DIR = Path(__file__).parent / "images"

paths = sorted(IMAGE_DIR.glob("*.jpg"))
print(f"Found {len(paths)} images")

sizes = {}
for path in paths:
    img = cv2.imread(str(path))
    if img is None:
        print(f"COULD NOT READ: {path.name}")
        continue
    height, width = img.shape[:2]
    sizes.setdefault((width, height), []).append(path.name)

for (width, height), names in sizes.items():
    print(f"{width} x {height}: {len(names)} images")