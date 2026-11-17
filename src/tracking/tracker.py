from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class Track:
    id: int
    box: tuple  # x1, y1, x2, y2


class PersonTracker:
    def __init__(self, weights="yolo11n.pt", conf=0.4, imgsz=640):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def update(self, frame):
        result = self.model.track(
            frame, persist=True, classes=[0], conf=self.conf,
            imgsz=self.imgsz, tracker="bytetrack.yaml", verbose=False,
        )[0]
        if result.boxes.id is None:
            return []
        return [
            Track(int(i), tuple(b))
            for i, b in zip(result.boxes.id, result.boxes.xyxy.tolist())
        ]

    def detect_once(self, frame):
        """Single image: no tracking, people numbered 1..n."""
        result = self.model.predict(
            frame, classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        return [
            Track(i, tuple(b))
            for i, b in enumerate(result.boxes.xyxy.tolist(), start=1)
        ]