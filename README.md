# PPE Detection

Detects helmets, hi-vis vests, and boots on people in images, video, or live camera. Flags people missing required PPE and saves evidence.

## How it works
1. `best.pt` (YOLO11s, trained on Roboflow PPE dataset v1) detects PPE items.
2. `yolo11n.pt` (COCO) detects people; ByteTrack gives each person an ID.
3. PPE boxes are assigned to the person they sit on (`src/compliance/association.py`).
4. A person missing a required item for N frames is a violation (`rules.py`).
5. Each violation saves a snapshot and a CSV row to `outputs/evidence/`.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Place trained weights at `models/trained/best.pt` (not in git).

## Run
```bash
python main.py --image path/to/photo.jpg
python main.py --video path/to/video.mp4
python main.py --camera 0
```
Press `q` to quit. Video runs save to `outputs/videos/`.

## Config (`configs/model.yaml`)
- `classes`: class id to name map. Class 2 is unconfirmed and ignored.
- `min_conf`: per-class confidence thresholds.
- `required`: PPE items every person must wear.
- `violation_frames`: consecutive frames missing before alert.

## Training
Done in Google Colab with Ultralytics 8.4.215. Dataset: Roboflow `ppe-detection-yj4rr` v1 (YOLOv11 export). 20 epochs. Test mAP50 0.974, mAP50-95 0.803.

## Tests
```bash
python -m pytest tests -v
```
