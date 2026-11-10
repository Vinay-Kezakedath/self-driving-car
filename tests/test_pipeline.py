"""Unit + integration tests. Run with:  pytest -q"""
import math

import numpy as np
import pytest

from src.config import AppConfig, DriveConfig
from src.decision import DrivingPlanner, State
from src.lane_detection import LaneDetector, LaneResult
from src.object_detection import ClassicalDetector, Detection, classify_traffic_light, estimate_distance
from src.pipeline import Pipeline
from src.road_simulator import RoadScene


@pytest.fixture(scope="module")
def scene():
    return RoadScene()


# --------------------------------------------------------------- geometry ---
def test_distance_estimate_pinhole():
    # a 1.85 m wide car that is 100 px wide with f=600 is 600*1.85/100 = 11.1 m away
    assert estimate_distance("car", (0, 0, 100, 50), 600) == pytest.approx(11.1)
    assert estimate_distance("unknown-thing", (0, 0, 10, 10), 600) == math.inf


def test_traffic_light_colour():
    import cv2
    for bgr, name in (((30, 30, 255), "red"), ((60, 230, 60), "green"), ((40, 190, 255), "amber")):
        img = np.zeros((40, 40, 3), np.uint8)
        cv2.circle(img, (20, 20), 8, bgr, -1)
        assert classify_traffic_light(img, (0, 0, 40, 40)) == name


# ------------------------------------------------------------------ lanes ---
def test_lane_detected_on_straight_road(scene):
    det = LaneDetector()
    res = None
    for _ in range(3):                                     # let the tracker settle
        res = det.process(scene.render(1.0))
    assert res.detected
    assert abs(res.lane_width_m - 3.5) < 0.3               # lane width recovered
    assert abs(res.offset_m - scene.truth(1.0)["offset_m"]) < 0.2


def test_curvature_sign_and_offset_accuracy(scene):
    """Measured curvature must follow the sign of the true curvature, and the
    vehicle offset must be within 15 cm on average over a stretch of video."""
    det = LaneDetector()
    errs, agree, total = [], 0, 0
    for i in range(0, 15 * 30, 6):
        t = i / 30
        r = det.process(scene.render(t))
        tr = scene.truth(t)
        if r.detected:
            errs.append(abs(r.offset_m - tr["offset_m"]))
            if abs(tr["curvature"]) > 0.002:               # clearly curved sections only
                total += 1
                agree += np.sign(r.curvature_m) == np.sign(tr["curvature"])
    assert np.mean(errs) < 0.15
    assert total > 0 and agree / total > 0.85


def test_lane_lost_on_blank_image():
    det = LaneDetector()
    res = det.process(np.full((720, 1280, 3), 90, np.uint8))
    assert not res.detected


# ----------------------------------------------------------- classic detect --
def test_classical_detector_finds_lead_car(scene):
    dets = ClassicalDetector().detect(scene.render(8.0))
    cars = [d for d in dets if d.label == "car"]
    assert cars
    true_z = scene.lead.zw(8.0, scene._integral) - scene._s_at(8.0)
    assert min(abs(c.distance_m - true_z) for c in cars) < 0.15 * true_z   # within 15 %


# ---------------------------------------------------------------- planner ---
def _lane(radius=math.inf, offset=0.0) -> LaneResult:
    """A fake perfect lane (straight, centred) for planner tests."""
    fit = np.array([0.0, 0.0, 520.0]), np.array([0.0, 0.0, 760.0])
    k = 0.0 if math.isinf(radius) else 1 / radius
    return LaneResult(True, fit[0], fit[1], curvature_m=k, radius_m=radius, offset_m=offset, lane_width_m=3.5)


def _planner():
    return DrivingPlanner(DriveConfig(), lane=None)        # lane=None -> corridor fallback test


def _car(dist, x=640):
    return Detection("car", 0.9, (x - 40, 400, x + 40, 450), dist)


def test_planner_cruise_when_clear():
    cmd = _planner().plan(_lane(), [], 1280)
    assert cmd.state == State.DRIVE


def test_planner_stops_for_close_obstacle():
    assert _planner().plan(_lane(), [_car(6)], 1280).state == State.STOP


def test_planner_slows_for_far_obstacle():
    assert _planner().plan(_lane(), [_car(20)], 1280).state == State.SLOW


def test_planner_ignores_car_in_other_lane():
    assert _planner().plan(_lane(), [_car(6, x=100)], 1280).state == State.DRIVE


def test_planner_stops_at_red_light():
    tl = Detection("traffic light", 0.9, (700, 300, 720, 340), 20, state="red")
    assert _planner().plan(_lane(), [tl], 1280).state == State.STOP
    tl.state = "green"
    assert _planner().plan(_lane(), [tl], 1280).state == State.DRIVE


def test_planner_slows_in_tight_curve():
    assert _planner().plan(_lane(radius=60), [], 1280).state == State.SLOW


def test_steering_direction():
    p = DrivingPlanner(DriveConfig(), None)
    p.speed_kmh = 40
    # car is 0.5 m LEFT of centre (offset -0.5) -> must steer RIGHT (+)
    assert p._steering(_lane(offset=-0.5)) > 0
    assert p._steering(_lane(offset=+0.5)) < 0
    # right-hand bend ahead -> steer right
    assert p._steering(_lane(radius=100)) > 0


# ------------------------------------------------------------ integration ---
def test_pipeline_end_to_end(scene):
    pipe = Pipeline(AppConfig(), (1280, 720), ClassicalDetector(), fps=30)
    out = None
    for t in (8.0, 8.1, 8.2):
        out = pipe.step(scene.render(t))
    assert out.annotated.shape == (720, 1280, 3)
    assert out.lane.detected
    assert out.command.state in (State.SLOW, State.STOP)    # lead car 17 m ahead
