"""Synthetic road-video generator.

Why does this exist?
    The project must "just work" without the user supplying footage, so we
    *render* a dash-cam style video with a simple pin-hole camera model:

      * a two-lane road (yellow left edge, dashed white divider, solid white
        right edge) that bends left and right,
      * grass, sky, trees, tree shadows and slow lighting changes (to stress
        the lane-detection thresholds),
      * a lead car that we catch up with, a car in the other lane,
      * a pedestrian who crosses the road in front of the car,
      * a traffic light that is red and later turns green.

How the geometry works (the key idea, read this first):
    The camera looks straight down the road. For every image row ``y`` below
    the horizon, a flat-ground pin-hole model gives the forward distance::

        z = focal_px * camera_height / (y - horizon_y)

    and for every column ``x`` the lateral position on the ground::

        X = (x - cx) * z / focal_px

    So for each pixel we know exactly where on the road it lies and can colour
    it (asphalt / line / grass). Objects (cars, trees...) are drawn as sprites
    scaled by ``focal_px / z``.

    The road shape is defined in *world* coordinates by a curvature profile
    kappa(Zw). The camera drives along the road and always looks along the road
    tangent, so a point ``z`` metres ahead appears shifted sideways by
    ``x_r(s+z) - x_r(s) - x_r'(s)*z  ~=  0.5 * kappa * z^2`` plus a small
    lane-keeping wobble (``drift``) that we also export as ground truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig

# ---------------------------------------------------------------- constants --
LANE_W = 3.5                 # lane width in metres
LEFT_EDGE = -1.5 * LANE_W    # yellow line  (X relative to ego-lane centre)
DIVIDER = -0.5 * LANE_W      # dashed white line (left boundary of ego lane)
RIGHT_EDGE = 0.5 * LANE_W    # solid white line  (right boundary of ego lane)

# BGR colours (OpenCV order!)
C_ASPHALT = np.array([88, 88, 90], np.float32)
C_GRASS = np.array([62, 128, 66], np.float32)
C_GRAVEL = np.array([118, 124, 128], np.float32)
C_WHITE = np.array([236, 236, 236], np.float32)
C_YELLOW = np.array([20, 195, 235], np.float32)


@dataclass
class Vehicle:
    """A car that lives in world coordinates and drives along the road."""

    lane_x: float     # lateral position relative to ego-lane centre (m)
    zw0: float        # world position at t=0 (m along the road)
    speed: float      # m/s (may be overridden by `speed_fn`)
    color: tuple      # BGR body colour
    speed_fn: object = None   # optional f(t) -> speed, for non-constant motion

    def zw(self, t: float, dt_table) -> float:
        """World position at time ``t`` (integrates speed_fn if present)."""
        if self.speed_fn is None:
            return self.zw0 + self.speed * t
        return self.zw0 + float(dt_table(self.speed_fn, t))


class RoadScene:
    """Renders frames of the synthetic drive. ``render(t)`` -> BGR image."""

    def __init__(self, cam: CameraConfig | None = None, fps: int = 30,
                 duration: float = 28.0, seed: int = 7):
        self.cam = cam or CameraConfig()
        self.fps = fps
        self.duration = duration
        self.rng = np.random.default_rng(seed)
        self.cx = self.cam.width / 2.0

        # ---- 1. ego speed profile v(t) -> travelled distance s(t) ----------
        # Pre-integrate on a fine time grid so s(t) is a cheap lookup later.
        self._tgrid = np.arange(0, duration + 1 / 240, 1 / 240)
        v = np.array([self._ego_speed(t) for t in self._tgrid])
        self._sgrid = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 * (1 / 240))])

        # ---- 2. road centre-line x_r(Zw) from a curvature profile ----------
        # kappa(Zw) is a sum of two sines -> gentle left/right bends.
        # Integrate twice (cumsum) to get heading and lateral position.
        self._zgrid = np.arange(0, 900, 0.25)
        kappa = 0.0040 * np.sin(2 * np.pi * self._zgrid / 240.0 + 0.6) \
            + 0.0015 * np.sin(2 * np.pi * self._zgrid / 95.0)
        # fade the curvature in so the video starts on a straight piece
        kappa *= np.clip(self._zgrid / 60.0, 0, 1)
        self._kappa = kappa
        self._heading = np.cumsum(kappa) * 0.25          # x_r'(Zw)
        self._xr = np.cumsum(self._heading) * 0.25       # x_r(Zw)

        # ---- 3. other actors ----------------------------------------------
        # Lead car in the ego lane: slower than us, so we close the gap and the
        # decision module has to react. It pulls away again after t = 12 s.
        self.lead = Vehicle(0.0, 62.0, 8.0, (40, 40, 200),
                            speed_fn=lambda t: 8.0 if t < 12 else min(8.0 + 2.5 * (t - 12), 17.0))
        # A car in the left lane that overtakes... then drifts away ahead.
        self.left_car = Vehicle(-LANE_W, 24.0, 17.0, (190, 110, 40))
        # Pedestrian crossing from the right-hand verge to the left.
        self.ped_zw = self._s_at(16.0) + 28.0

        # Trees on both sides every ~16 m (pseudo-random lateral offsets).
        n_trees = 60
        self.trees = [
            (10.0 + 16.0 * i + self.rng.uniform(-3, 3),            # world Z
             (1 if i % 2 else -1) * self.rng.uniform(7.0, 12.0)      # lateral X
             - (LANE_W if i % 2 == 0 else 0.0),
             self.rng.uniform(5.5, 8.5))                              # height m
            for i in range(n_trees)
        ]
        # Traffic light fixed in the world on the right of the road.
        self.tl_zw = 235.0
        self.tl_x = RIGHT_EDGE + 2.2

    # ------------------------------------------------------------- dynamics --
    @staticmethod
    def _ego_speed(t: float) -> float:
        """Piece-wise linear ego speed (m/s): cruise, follow lead, stop, go."""
        pts = [(0, 14), (6, 14), (9, 8), (16, 8), (20, 4), (24, 4), (28, 12)]
        ts, vs = zip(*pts)
        return float(np.interp(t, ts, vs))

    def _s_at(self, t: float) -> float:
        """Distance travelled by the ego car at time t (m)."""
        return float(np.interp(t, self._tgrid, self._sgrid))

    def _integral(self, fn, t: float) -> float:
        """Numerically integrate a speed function from 0..t (trapezoid)."""
        ts = np.linspace(0, t, max(2, int(t * 60)))
        vals = np.array([fn(x) for x in ts])      # manual trapezoid: works on NumPy 1.x and 2.x
        return float(np.sum((vals[1:] + vals[:-1]) / 2 * np.diff(ts)))

    def _curve_rel(self, s: float, z: np.ndarray) -> np.ndarray:
        """Sideways shift (m) of the road centre-line at distance ``z`` ahead,
        relative to the straight line the camera is looking along."""
        zi = np.clip(s + z, 0, self._zgrid[-1])
        xr_ahead = np.interp(zi, self._zgrid, self._xr)
        xr_here = np.interp(s, self._zgrid, self._xr)
        head_here = np.interp(s, self._zgrid, self._heading)
        return xr_ahead - xr_here - head_here * z

    def drift(self, t: float) -> float:
        """Lane-keeping wobble: camera offset from lane centre (m, +right)."""
        return 0.30 * np.sin(0.55 * t) + 0.10 * np.sin(1.7 * t + 1)

    def truth(self, t: float) -> dict:
        """Ground truth for evaluation: signed curvature (1/m, + = right bend),
        vehicle offset from the lane centre (m, + = car right of centre)."""
        s = self._s_at(t)
        # curvature a few metres ahead (matches what the detector measures)
        # NOTE: in the renderer a positive x_r shifts the road to the LEFT on
        # screen, so the physical (right-positive) curvature is the negative.
        k = -float(np.interp(s + 6.0, self._zgrid, self._kappa))
        return {"t": t, "curvature": k, "offset_m": float(self.drift(t)),
                "ego_speed_ms": self._ego_speed(t)}

    # ------------------------------------------------------------- rendering --
    def render(self, t: float) -> np.ndarray:
        cam = self.cam
        W, H, hor, f = cam.width, cam.height, cam.horizon_y, cam.focal_px
        s = self._s_at(t)
        drift = self.drift(t)

        # ---------- sky + distant hills ---------------------------------
        frame = np.empty((H, W, 3), np.float32)
        g = np.linspace(0, 1, hor, dtype=np.float32)[:, None]
        sky_top, sky_hor = np.array([225, 170, 100], np.float32), np.array([250, 238, 225], np.float32)
        frame[:hor] = (sky_top * (1 - g) + sky_hor * g)[:, None, :]
        xs = np.arange(W, dtype=np.float32)
        hills = hor - 22 - 14 * np.sin(xs / 190.0 + 1) - 8 * np.sin(xs / 61.0)
        for x in range(W):                      # paint hill silhouette column-wise
            frame[int(hills[x]):hor, x] = (95, 120, 95)

        # ---------- ground: per-pixel world position ----------------------
        rows = np.arange(hor, H)
        dy = (rows - hor + 0.5).astype(np.float32)
        z = f * cam.height_m / dy                          # forward distance per row
        rel = self._curve_rel(s, z).astype(np.float32)      # road bend per row
        # lane-space lateral coordinate X for every pixel of every ground row
        X = (xs[None, :] - self.cx) * z[:, None] / f + drift + rel[:, None]
        px_w = (z / f)[:, None]                             # metres per pixel (for anti-aliasing)

        # base colours: grass everywhere, gravel shoulder, asphalt road
        ground = np.broadcast_to(C_GRASS, (len(rows), W, 3)).copy()
        gravel = (X > RIGHT_EDGE) & (X < RIGHT_EDGE + 1.2) | (X < LEFT_EDGE) & (X > LEFT_EDGE - 1.2)
        ground[gravel] = C_GRAVEL
        road = (X >= LEFT_EDGE) & (X <= RIGHT_EDGE)
        ground[road] = C_ASPHALT

        # lane markings, anti-aliased by pixel footprint; min width keeps far lines visible
        wl = np.maximum(0.15, 1.3 * z / f)[:, None]
        zw = (s + z)[:, None]                               # world distance per row
        def line(x0, colour, mask=1.0):
            a = np.clip((wl / 2 - np.abs(X - x0)) / px_w + 0.5, 0, 1) * mask
            ground[:] = ground * (1 - a[..., None]) + colour * a[..., None]
        line(LEFT_EDGE + 0.1, C_YELLOW)                    # solid yellow
        line(RIGHT_EDGE - 0.1, C_WHITE)                    # solid white
        dash = ((zw % 9.0) < 3.6).astype(np.float32)       # 3.6 m dash / 5.4 m gap
        line(DIVIDER, C_WHITE, dash)                       # dashed white

        # texture noise so thresholds are not trivially clean
        ground += self.rng.normal(0, 4.0, ground.shape).astype(np.float32)

        # tree shadows lying across the road + slow global light change
        shadow = ((zw + 7.0) % 41.0 < 7.0)
        ground[(shadow & road)] *= 0.62
        frame[hor:] = ground

        # ---------- sprites, painted far -> near (painter's algorithm) -----
        items = []   # (z, draw_fn)
        for tz, tx, th in self.trees:
            zt = tz - s
            if 8 < zt < 160:
                items.append((zt, lambda zt=zt, tx=tx, th=th: self._tree(frame, s, zt, tx, th, drift)))
        zl = self.lead.zw(t, self._integral) - s
        if 3 < zl < 160:
            items.append((zl, lambda: self._car(frame, s, zl, 0.0, self.lead.color, drift, brake=t < 12 and zl < 30)))
        zc = self.left_car.zw(t, self._integral) - s
        if 3 < zc < 160:
            items.append((zc, lambda: self._car(frame, s, zc, -LANE_W, self.left_car.color, drift)))
        zp = self.ped_zw - s
        if 3 < zp < 120:
            px_x = 4.5 - 1.4 * np.clip(t - 15.0, 0, 6.2)      # walks 4.5 m -> -4.2 m
            items.append((zp, lambda: self._pedestrian(frame, s, zp, px_x, drift, t)))
        ztl = self.tl_zw - s
        if 3 < ztl < 200:
            red = t < 26.0
            items.append((ztl, lambda: self._traffic_light(frame, s, ztl, red, drift)))
        for _, fn in sorted(items, key=lambda it: -it[0]):
            fn()

        # ---------- camera effects: exposure, blur, sensor noise ----------
        gain = 1.0 - 0.12 * np.sin(2 * np.pi * t / 20.0) ** 2
        frame *= gain
        out = np.clip(frame, 0, 255).astype(np.uint8)
        out = cv2.GaussianBlur(out, (3, 3), 0.8)
        noise = self.rng.normal(0, 3.0, out.shape)
        return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # --------------------------------------------------------- sprite helpers --
    def _project(self, s, z, lane_x, drift):
        """World (z ahead, lateral lane_x) -> screen (x, ground-y, metres->px scale)."""
        rel = float(self._curve_rel(s, np.array([z]))[0])
        scale = self.cam.focal_px / z
        x = self.cx + scale * (lane_x - drift - rel)
        y = self.cam.horizon_y + self.cam.height_m * scale
        return x, y, scale

    def _tree(self, img, s, z, lane_x, h_m, drift):
        x, y, sc = self._project(s, z, lane_x, drift)
        trunk_w = max(1, int(0.4 * sc))
        cv2.rectangle(img, (int(x - trunk_w / 2), int(y - 0.4 * h_m * sc)), (int(x + trunk_w / 2), int(y)), (40, 70, 100), -1)
        cv2.ellipse(img, (int(x), int(y - 0.7 * h_m * sc)), (int(0.22 * h_m * sc), int(0.32 * h_m * sc)), 0, 0, 360, (45, 105, 50), -1)

    def _car(self, img, s, z, lane_x, colour, drift, brake=False):
        """Rear view of a car: shadow, body, cabin + window, tail lights, plate, wheels."""
        x, y, sc = self._project(s, z, lane_x, drift)
        w, h = 1.85 * sc, 1.5 * sc
        if w < 3:
            return
        x0, x1 = int(x - w / 2), int(x + w / 2)
        cv2.ellipse(img, (int(x), int(y)), (int(w * 0.62), max(1, int(h * 0.06))), 0, 0, 360, (35, 35, 35), -1)   # ground shadow
        cv2.rectangle(img, (x0, int(y - 0.62 * h)), (x1, int(y - 0.10 * h)), colour, -1)                         # lower body
        cv2.rectangle(img, (int(x - 0.40 * w), int(y - h)), (int(x + 0.40 * w), int(y - 0.60 * h)), colour, -1)  # cabin
        cv2.rectangle(img, (int(x - 0.34 * w), int(y - 0.94 * h)), (int(x + 0.34 * w), int(y - 0.66 * h)), (70, 60, 55), -1)  # rear window
        tail = (30, 30, 255) if brake else (20, 20, 150)
        lw, lh = max(2, int(0.2 * w)), max(2, int(0.11 * h))
        for sx in (x0 + 2, x1 - 2 - lw):                                                                          # tail lights
            cv2.rectangle(img, (sx, int(y - 0.50 * h)), (sx + lw, int(y - 0.50 * h) + lh), tail, -1)
        cv2.rectangle(img, (int(x - 0.14 * w), int(y - 0.36 * h)), (int(x + 0.14 * w), int(y - 0.28 * h)), (225, 225, 225), -1)  # plate
        cv2.rectangle(img, (x0, int(y - 0.14 * h)), (x1, int(y - 0.06 * h)), (30, 30, 30), -1)                    # bumper
        ww = max(2, int(0.16 * w))
        for sx in (x0 + 1, x1 - 1 - ww):                                                                          # wheels
            cv2.rectangle(img, (sx, int(y - 0.22 * h)), (sx + ww, int(y)), (20, 20, 20), -1)

    def _pedestrian(self, img, s, z, lane_x, drift, t):
        """A walking person: head, torso, swinging legs and arms (1.7 m tall)."""
        x, y, sc = self._project(s, z, lane_x, drift)
        h = 1.7 * sc
        if h < 6:
            return
        swing = np.sin(t * 7.0) * 0.12 * h
        cx = int(x)
        cv2.circle(img, (cx, int(y - 0.92 * h)), max(2, int(0.07 * h)), (150, 190, 230), -1)               # head
        cv2.rectangle(img, (int(cx - 0.13 * h), int(y - 0.85 * h)), (int(cx + 0.13 * h), int(y - 0.48 * h)), (170, 150, 30), -1)  # torso (teal, distinct from the red cars so blobs do not merge)
        lw = max(2, int(0.07 * h))
        cv2.line(img, (cx - lw, int(y - 0.48 * h)), (int(cx - lw + swing), int(y)), (85, 80, 80), lw)      # legs
        cv2.line(img, (cx + lw, int(y - 0.48 * h)), (int(cx + lw - swing), int(y)), (85, 80, 80), lw)
        cv2.line(img, (int(cx - 0.14 * h), int(y - 0.82 * h)), (int(cx - 0.14 * h - swing * .6), int(y - 0.52 * h)), (170, 150, 30), lw)  # arms
        cv2.line(img, (int(cx + 0.14 * h), int(y - 0.82 * h)), (int(cx + 0.14 * h + swing * .6), int(y - 0.52 * h)), (170, 150, 30), lw)

    def _traffic_light(self, img, s, z, red, drift):
        """Pole + black housing with red / amber / green lamps (one lit)."""
        x, y, sc = self._project(s, z, self.tl_x, drift)
        top = int(y - 4.6 * sc)
        cv2.line(img, (int(x), int(y)), (int(x), top), (90, 90, 90), max(1, int(0.12 * sc)))
        bw, bh = 0.45 * sc, 1.15 * sc
        cv2.rectangle(img, (int(x - bw / 2), top), (int(x + bw / 2), int(top + bh)), (25, 25, 25), -1)
        lamp_r = max(1, int(0.13 * sc))
        lamps = [((30, 30, 255), red), ((40, 190, 255), False), ((60, 230, 60), not red)]   # BGR red, amber, green
        for i, (col, on) in enumerate(lamps):
            c = col if on else tuple(int(v * 0.18) for v in col)
            cv2.circle(img, (int(x), int(top + bh * (0.2 + 0.3 * i))), lamp_r, c, -1)


def generate(out_path: str | Path, duration: float = 28.0, fps: int = 30, seed: int = 7) -> Path:
    """Render the full drive to ``out_path`` (mp4) and write ``*.truth.json``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = RoadScene(fps=fps, duration=duration, seed=seed)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (scene.cam.width, scene.cam.height))
    truth = []
    for i in range(int(duration * fps)):
        t = i / fps
        writer.write(scene.render(t))
        truth.append(scene.truth(t))
    writer.release()
    out_path.with_suffix(".truth.json").write_text(json.dumps(truth))
    return out_path
