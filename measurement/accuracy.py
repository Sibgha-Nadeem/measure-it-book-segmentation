import csv
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
import measure

TRUE_HEIGHT_MM = 198.12
TRUE_WIDTH_MM = 129.54
TEST_RANGE = range(91, 101)
NAME_MAP = ROOT / "dataset" / "splits" / "name_map.csv"
RAW_DIR = ROOT / "dataset" / "raw"
OUT_CSV = ROOT / "measurement" / "accuracy_results.csv"
OUT_LOG = ROOT / "measurement" / "accuracy_log.txt"


def test_photos():
    pairs = []
    with open(NAME_MAP, newline="") as f:
        for row in csv.reader(f):
            book = next((c for c in row if re.search(r"book_\d{3}", c)), None)
            others = [c for c in row if c and c != book]
            if book is None or not others:
                continue
            number = int(re.search(r"book_(\d{3})", book).group(1))
            if number in TEST_RANGE:
                pairs.append((f"book_{number:03d}", Path(others[0]).name))
    return sorted(pairs)


def measure_one(model, params, path, undistort):
    image = cv2.imread(str(path))
    if image is None:
        return None, "unreadable"
    if undistort:
        image = cv2.undistort(image, params["camera_matrix"], params["dist_coeffs"])
    mask, confidence = measure.segment_book(model, image)
    if mask is None:
        return None, "book not found"
    homography, _ = measure.board_homography(image)
    if homography is None:
        return None, "reference not found"
    if confidence < measure.MIN_CONFIDENCE:
        return None, "low confidence"
    height_mm, width_mm = measure.measure_mm(measure.book_corners(mask), homography)
    return (width_mm, height_mm, confidence), "ok"


def run_mode(model, params, pairs, undistort):
    rows = []
    mode = "undistorted" if undistort else "raw"
    for book, original in pairs:
        row = {"mode": mode, "book": book, "photo": original, "status": "photo missing"}
        path = next(RAW_DIR.rglob(original), None)
        if path is not None:
            start = time.perf_counter()
            result, row["status"] = measure_one(model, params, path, undistort)
            row["seconds"] = time.perf_counter() - start
            if result is not None:
                width, height, confidence = result
                row.update(
                    width_mm=width,
                    height_mm=height,
                    confidence=confidence,
                    width_err=width - TRUE_WIDTH_MM,
                    height_err=height - TRUE_HEIGHT_MM,
                )
        rows.append(row)
    return rows


def report(rows):
    mode = rows[0]["mode"]
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [f"== {mode}: {len(ok)} measured, {len(rows) - len(ok)} not measured =="]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"{r['book']} | {r['status']}")
            continue
        lines.append(
            f"{r['book']} | W {r['width_mm']:.2f} ({r['width_err']:+.2f}) | "
            f"H {r['height_mm']:.2f} ({r['height_err']:+.2f}) | "
            f"conf {r['confidence']:.3f} | {r['seconds']:.2f} s"
        )
    if ok:
        abs_errors = [abs(r[k]) for r in ok for k in ("width_err", "height_err")]
        pct_errors = [abs(r["width_err"]) / TRUE_WIDTH_MM * 100 for r in ok] + [
            abs(r["height_err"]) / TRUE_HEIGHT_MM * 100 for r in ok
        ]
        lines.append(
            f"MAE {statistics.mean(abs_errors):.2f} mm | median {statistics.median(abs_errors):.2f} mm | "
            f"worst {max(abs_errors):.2f} mm | mean % error {statistics.mean(pct_errors):.2f}"
        )
    return lines


def main():
    pairs = test_photos()
    print(f"Test photos found in name_map.csv: {len(pairs)}")
    params = np.load(measure.PARAMS_PATH)
    model = measure.load_model()
    all_rows, lines = [], []
    for undistort in (True, False):
        rows = run_mode(model, params, pairs, undistort)
        all_rows += rows
        lines += report(rows) + [""]
    print("\n".join(lines))
    OUT_LOG.write_text("\n".join(lines))
    columns = ["mode", "book", "photo", "status", "width_mm", "height_mm",
               "width_err", "height_err", "confidence", "seconds"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()