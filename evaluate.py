"""Quantitative check of the lane module against the simulator's ground truth.

    python evaluate.py            # uses data/sample_road.mp4 + .truth.json

Reports how well the *measured* vehicle offset and road curvature match the
values the simulator used to render each frame.
"""
import json
import sys

import cv2
import numpy as np

from src.config import LaneConfig
from src.lane_detection import LaneDetector

video = sys.argv[1] if len(sys.argv) > 1 else "data/sample_road.mp4"
truth = json.load(open(video.replace(".mp4", ".truth.json")))
cap = cv2.VideoCapture(video)
det = LaneDetector(LaneConfig(), (int(cap.get(3)), int(cap.get(4))))

off_err, curv_meas, curv_true, found = [], [], [], 0
for tr in truth:
    ok, frame = cap.read()
    if not ok:
        break
    r = det.process(frame)
    if r.detected:
        found += 1
        off_err.append(r.offset_m - tr["offset_m"])
        curv_meas.append(r.curvature_m)
        curv_true.append(tr["curvature"])

off_err, cm, ct = map(np.array, (off_err, curv_meas, curv_true))
print(f"frames                : {len(truth)}")
print(f"lane detected         : {100 * found / len(truth):.1f} %")
print(f"offset error  (MAE)   : {np.abs(off_err).mean() * 100:.1f} cm   (max {np.abs(off_err).max() * 100:.1f} cm)")
print(f"curvature corr. (r)   : {np.corrcoef(cm, ct)[0, 1]:.3f}")
print(f"curvature error (MAE) : {np.abs(cm - ct).mean():.5f} 1/m   (true range {ct.min():+.4f}..{ct.max():+.4f})")
