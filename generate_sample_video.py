"""CLI: render the synthetic sample road video into data/sample_road.mp4."""
import argparse

from src.road_simulator import generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sample_road.mp4")
    ap.add_argument("--duration", type=float, default=28.0, help="seconds")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print("wrote", generate(a.out, a.duration, a.fps, a.seed))
