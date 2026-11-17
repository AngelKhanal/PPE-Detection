from dataclasses import dataclass
from pathlib import Path

import yaml
from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple  # x1, y1, x2, y2 in pixels


class Detector:
    def __init__(self, config_path="configs/model.yaml"):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.model = YOLO(cfg["weights"])
        self.conf = cfg["conf"]
        self.imgsz = cfg["imgsz"]
        self.classes = cfg["classes"]
        self.min_conf = cfg.get("min_conf", {})

    def detect(self, frame):
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        detections = [
            Detection(self.classes[int(c)], float(p), tuple(b))
            for c, p, b in zip(
                result.boxes.cls, result.boxes.conf, result.boxes.xyxy.tolist()
            )
        ]
        return [d for d in detections if d.conf >= self.min_conf.get(d.label, self.conf)]