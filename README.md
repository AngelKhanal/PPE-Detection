# PPE Detection

Detects helmets, hi-vis vests, and boots on people in images, video files, webcams, and CCTV streams. Flags people missing required PPE, shows a per-person ID, and saves evidence.

## How it works
1. `best.pt` (YOLO11s, trained on Roboflow PPE dataset) detects PPE items: helmet, boots, vest.
2. `yolo11n.pt` (COCO, auto-downloaded) detects people. ByteTrack gives each person an ID in video and live modes.
3. Each PPE box is assigned to the person it sits on (`src/compliance/association.py`): helmet near the head, vest on the torso, boots near the feet.
4. A person missing a required item for N consecutive frames is a violation (`src/compliance/rules.py`). Image mode flags on one frame.
5. Each new violation saves a snapshot and one CSV row to `outputs/evidence/` (`src/compliance/evidence.py`).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Put the trained weights at `models/trained/best.pt`. Weights are not in git.

## Run
```bash
python main.py --image path/to/photo.jpg      # photo, saved to outputs/images/
python main.py --video path/to/video.mp4      # video file, saved to outputs/videos/
python main.py --camera 0                     # webcam (try 1 for another camera)
python main.py --stream "rtsp://..."          # any RTSP/HTTP stream
python main.py --cctv                         # stream URL from configs/cctv.local.yaml
```
Press `q` in the window to quit. Run all commands from the project root.

Optional: `--out path` sets the output file for image and video modes.

Image formats: jpg, png, webp, and avif (via Pillow fallback).

### Output
- Green person box `ID n OK`: all required PPE found.
- Red person box `ID n MISSING: helmet, vest`: violation.
- Image mode IDs are just 1..n. In video and live modes, IDs are tracked, so the same person keeps the same ID while in view. A person who leaves and returns may get a new ID. IDs reset on every restart.
- Evidence: `outputs/evidence/` holds snapshots and `violations.csv` (time, track_id, missing, snapshot). One row per new violation per person.

## CCTV setup
1. Check the stream plays in VLC (Media > Open Network Stream). If VLC cannot play it, this program cannot either.
2. Copy `configs/cctv.example.yaml` to `configs/cctv.local.yaml` and set `url`.
3. Run `python main.py --cctv`.

`configs/*.local.yaml` is gitignored. Never commit stream URLs, they contain passwords. Prefer the camera's sub-stream (lower resolution): faster, and the model only uses 640 px anyway.

The reader thread always processes the newest frame, so the view does not lag behind live. The stream reconnects automatically if it drops.

## Config (`configs/model.yaml`)
- `classes`: class id to name map for `best.pt`. Class 2 is a whole-body person box, hidden in drawing and not used for rules (the dataset labels it inconsistently).
- `min_conf`: per-class confidence thresholds, applied after detection.
- `person`: weights and confidence for the person detector.
- `required`: PPE items every person must wear.
- `violation_frames`: consecutive frames a person must be missing an item before it counts as a violation.

Display size and text size: `fit_to_screen` and `TEXT_SCALE` in `src/visualization/draw_results.py`. These affect only the on-screen window and overlays, not detection.

## Project structure
```
configs/        model.yaml, cctv.example.yaml (cctv.local.yaml is gitignored)
src/detection/  detector.py: PPE detector
src/tracking/   tracker.py: person detection and tracking
src/compliance/ association.py, rules.py, evidence.py
src/video/      image_source.py, video_source.py, camera_source.py
src/visualization/ draw_results.py
tests/          unit tests for association, rules, evidence
main.py         command line entry point
```
Folders `datasets/`, `models/`, `outputs/` are gitignored.

## Training
Done in Google Colab, then the weights are downloaded.
- Ultralytics 8.4.215, model `yolo11s.pt`, 20 epochs, image size 640, Tesla T4 GPU.
- Dataset: Roboflow Universe `ppe-detection-yj4rr` version 1 (YOLOv11 export): 788 train, 226 valid, 112 test images.
- Test set: mAP50 0.974, mAP50-95 0.803. Per class mAP50-95: helmet 0.751, boots 0.732, person 0.900, vest 0.828.

Use the same Ultralytics version locally as in Colab (pinned in `requirements.txt`).

## Tests
```bash
python -m pytest tests -v
```

- The dataset license on Roboflow Universe should be checked before any commercial use.