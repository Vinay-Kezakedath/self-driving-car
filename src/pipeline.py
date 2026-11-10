"""Glue: video source -> lane detection -> object detection -> planner -> output.

``Pipeline.step(frame)`` is the only place where the modules are wired together;
everything else (CLI, tests, notebooks) just calls it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import AppConfig
from .decision import Command, DrivingPlanner
from .lane_detection import LaneDetector, LaneResult
from .object_detection import Detection, Detector
from .visualization import render


@dataclass
class FrameResult:
    lane: LaneResult
    detections: list[Detection]
    command: Command
    annotated: np.ndarray
    fps: float


class Pipeline:
    def __init__(self, cfg: AppConfig, frame_size: tuple[int, int], detector: Detector | None,
                 fps: float = 30.0, detect_every: int = 1):
        self.cfg = cfg
        self.lane = LaneDetector(cfg.lane, frame_size)
        self.planner = DrivingPlanner(cfg.drive, self.lane, fps)
        self.detector = detector
        self.frame_w = frame_size[0]
        self.detect_every = max(1, detect_every)   # run the (slow) detector every N frames
        self._n = 0
        self._last_dets: list[Detection] = []
        self._fps = 0.0                             # smoothed processing speed

    def step(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        lane = self.lane.process(frame)                             # 1. where is the lane?
        if self.detector is not None and self._n % self.detect_every == 0:
            self._last_dets = self.detector.detect(frame)           # 2. what is on the road?
        self._n += 1
        cmd = self.planner.plan(lane, self._last_dets, self.frame_w)  # 3. what should we do?
        dt = time.perf_counter() - t0
        self._fps = 1.0 / dt if self._fps == 0 else 0.9 * self._fps + 0.1 / dt
        out = render(frame, self.lane, lane, self._last_dets, cmd, self._fps)   # 4. draw
        return FrameResult(lane, self._last_dets, cmd, out, self._fps)
