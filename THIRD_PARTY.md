# Third-party material

This project's own code is licensed under AGPL-3.0 (see `LICENSE`). It uses the
following third-party software and data, which keep their own licences.

## Software

| Component | Licence | Notes |
|---|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`ultralytics` library and `yolov8n.pt` weights) | AGPL-3.0 (commercial licence available from Ultralytics) | Optional. Not stored in this repo: install via `requirements.txt` and download the weights as described in `README.md`. |
| [OpenCV](https://opencv.org/) (`opencv-python`) | Apache-2.0 | |
| [NumPy](https://numpy.org/) | BSD-3-Clause | |
| [PyTorch / torchvision](https://pytorch.org/) | BSD-3-Clause | Only needed for YOLO. |
| [pytest](https://pytest.org/) | MIT | Tests only. |

## Video footage (not stored in this repo)

The optional demo videos in `data/real/` come from Udacity's public repositories,
released under the MIT licence. Copyright (c) Udacity, Inc.

| File | Source |
|---|---|
| `project_video.mp4`, `challenge_video.mp4`, `harder_challenge_video.mp4` | https://github.com/udacity/CarND-Advanced-Lane-Lines |
| `challenge_p1.mp4` (`test_videos/challenge.mp4`), `solid_white_right.mp4` (`solidWhiteRight.mp4`), `solid_yellow_left.mp4` (`solidYellowLeft.mp4`) | https://github.com/udacity/CarND-LaneLines-P1 |

MIT licence text: see the `LICENSE` file in each of the repositories above.
