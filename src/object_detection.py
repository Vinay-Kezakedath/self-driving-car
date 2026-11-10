"""Object detection: cars, pedestrians, traffic lights, stop signs.

Two interchangeable back-ends implement the same ``detect(frame)`` interface,
so the rest of the pipeline never needs to know which one is running:

``YoloDetector``
    Deep-learning detector (Ultralytics YOLOv8-nano, COCO weights, CPU is fine).
    Use this on REAL dash-cam footage.

``ClassicalDetector``
    A tiny, hand-written OpenCV colour/blob detector. The synthetic sample
    video is made of flat-shaded cartoon sprites which a COCO-trained network
    cannot recognise (it misses the cars and hallucinates stop signs), so the
    demo video uses this detector instead. It is deliberately simple and only
    understands the sprites drawn by ``road_simulator.py``.

Both produce :class:`Detection` objects that already carry a *distance estimate*
from the pin-hole model::

        distance = focal_px * real_size_m / size_in_pixels
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

# Real-world sizes (metres) used for monocular distance estimation.
# (dimension, size): we use the box WIDTH for vehicles and HEIGHT for upright things.
REAL_SIZE = {
    "car": ("w", 1.85), "truck": ("w", 2.5), "bus": ("w", 2.55),
    "motorcycle": ("w", 0.8), "bicycle": ("w", 0.6),
    "person": ("h", 1.7), "traffic light": ("h", 1.15), "stop sign": ("h", 0.75),
}
VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}


@dataclass
class Detection:
    """One detected object in image coordinates."""

    label: str
    conf: float
    box: tuple[int, int, int, int]        # x1, y1, x2, y2 (pixels)
    distance_m: float = float("inf")      # monocular range estimate
    state: str | None = None              # traffic lights: "red" | "amber" | "green" | "unknown"

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Where the object touches the ground (used for lane membership tests)."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, float(y2))


class Detector(Protocol):
    """Structural interface every back-end satisfies."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def estimate_distance(label: str, box, focal_px: float) -> float:
    """Pin-hole range estimate from the apparent size of the box."""
    if label not in REAL_SIZE:
        return float("inf")
    dim, size = REAL_SIZE[label]
    x1, y1, x2, y2 = box
    px = (x2 - x1) if dim == "w" else (y2 - y1)
    return float(focal_px * size / px) if px > 0 else float("inf")


def classify_traffic_light(frame: np.ndarray, box) -> str:
    """Decide red / amber / green by counting bright, saturated pixels of each
    hue inside the box (HSV makes colour tests independent of brightness)."""
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 150)          # lit & coloured
    h = hsv[:, :, 0]
    counts = {
        "red": int((bright & ((h < 10) | (h > 165))).sum()),     # red wraps around 0/180
        "amber": int((bright & (h >= 10) & (h < 35)).sum()),
        "green": int((bright & (h >= 40) & (h < 90)).sum()),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] >= 2 else "unknown"


# --------------------------------------------------------------------- YOLO --
class YoloDetector:
    """Ultralytics YOLO wrapper restricted to driving-relevant COCO classes."""

    WANTED = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"}

    def __init__(self, weights: str = "models/yolov8n.pt", conf: float = 0.35,
                 focal_px: float = 600.0, imgsz: int = 640, device: str = "cpu"):
        # Imported lazily: users who only want the classical detector do not
        # need PyTorch installed at all.
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.focal_px, self.imgsz, self.device = conf, focal_px, imgsz, device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out: list[Detection] = []
        for b in res.boxes:
            label = self.model.names[int(b.cls)]
            if label not in self.WANTED:
                continue
            box = tuple(int(v) for v in b.xyxy[0].tolist())
            d = Detection(label, float(b.conf), box, estimate_distance(label, box, self.focal_px))
            if label == "traffic light":
                d.state = classify_traffic_light(frame, box)
            out.append(d)
        return out


# ---------------------------------------------------------------- classical --
class ClassicalDetector:
    """Colour-blob detector for the synthetic sprites (see module docstring).

    Steps: HSV threshold -> morphological clean-up -> connected components ->
    shape rules (aspect ratio / size) -> Detection.
    """

    def __init__(self, focal_px: float = 600.0, horizon_y: int = 380):
        self.focal_px, self.horizon_y = focal_px, horizon_y
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def _components(self, mask: np.ndarray, min_area: int):
        """Connected components -> list of (x, y, w, h, fill) where ``fill`` is
        the fraction of the bounding box that is actually painted (compact
        blobs ~1.0, thin compression-artifact slivers are much lower)."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return [(*map(int, stats[i][:4]), stats[i][4] / max(1, stats[i][2] * stats[i][3]))
                for i in range(1, n) if stats[i][4] >= min_area]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # only look at the road scene: from slightly above the horizon downward
        roi = np.zeros((H, W), np.uint8)
        roi[max(0, self.horizon_y - 70):, :] = 255
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        red = (((h < 8) | (h > 172)) & (s > 120) & (v > 90)).astype(np.uint8) * 255 & roi
        blue = ((h > 102) & (h < 125) & (s > 150) & (v > 90)).astype(np.uint8) * 255 & roi
        teal = ((h > 82) & (h < 100) & (s > 120) & (v > 90)).astype(np.uint8) * 255 & roi   # pedestrian shirt
        dets: list[Detection] = []

        # --- cars: wide red / blue blobs (body + cabin). Tiny blobs (< 10 px
        #     wide, i.e. > ~110 m away or just tail lights) are ignored. -------
        for colour_mask in (red, blue):
            for x, y, w, hh, _fill in self._components(colour_mask, min_area=40):
                if w >= 10 and w / hh >= 0.9:
                    box = (x, y, x + w, y + int(hh * 1.12))   # extend down to bumper/wheels
                    dets.append(Detection("car", 0.9, box, estimate_distance("car", box, self.focal_px)))

        # --- pedestrians: teal torso blob; extrapolate the full body box ------
        for x, y, w, hh, fill in self._components(teal, min_area=30):
            # compact torso-shaped blob only: rejects codec fringes on car edges
            if hh >= 8 and w >= 5 and 0.5 < w / hh < 1.6 and fill > 0.6:
                # torso spans ~37 % of body height and starts ~15 % below the head top
                H_p = hh / 0.37
                top = y - 0.15 * H_p
                box = (int(x - 0.1 * w), int(top), int(x + 1.1 * w), int(top + H_p))
                dets.append(Detection("person", 0.85, box, estimate_distance("person", box, self.focal_px)))

        # --- traffic lights: a very bright, saturated red/green dot surrounded
        #     by a near-black housing (this separates it from red cars). --------
        lit = (((h < 8) | (h > 172) | ((h > 45) & (h < 85))) & (s > 150) & (v > 170)).astype(np.uint8) * 255   # whole frame: lights hang high above the horizon
        for x, y, w, hh, _fill in self._components(lit, min_area=3):
            if not (0.6 < w / hh < 1.6) or w > 40:
                continue
            # The unlit lamps / black housing sit directly above or below a lit
            # lamp. Check a strip below AND above for darkness (blur makes a
            # ring right around the lamp unreliable, so we skip its glow).
            g0, g1 = int(0.4 * w) + 1, int(1.3 * w) + 2
            below = hsv[y + hh + g0: y + hh + g1, x: x + w, 2]
            above = hsv[max(0, y - g1): max(0, y - g0), x: x + w, 2]
            if (below.size and below.mean() < 80) or (above.size and above.mean() < 80):
                # lamp diameter 0.26 m -> real distance; housing box ~ 1.7 x 4.4 lamp sizes
                box = (int(x - 0.4 * w), int(y - 0.6 * w), int(x + 1.4 * w), int(y + 3.8 * w))
                # +1 px: thresholding only keeps the bright core, so the lamp looks ~1 px too small
                d = Detection("traffic light", 0.8, box, self.focal_px * 0.26 / (w + 1))
                d.state = classify_traffic_light(frame, (x, y, x + w, y + hh))
                dets.append(d)
        return dets


def make_detector(kind: str, weights: str, conf: float, focal_px: float, horizon_y: int) -> Detector | None:
    """Factory used by the CLI. ``kind``: yolo | classical | none."""
    if kind == "yolo":
        return YoloDetector(weights, conf, focal_px)
    if kind == "classical":
        return ClassicalDetector(focal_px, horizon_y)
    return None
