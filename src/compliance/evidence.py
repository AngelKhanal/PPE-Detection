import csv
from datetime import datetime
from pathlib import Path

import cv2


class EvidenceLogger:
    def __init__(self, folder="outputs/evidence"):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.folder / "violations.csv"
        self.logged = {}  # track_id -> frozenset of items already logged
        if not self.csv_path.exists():
            self.csv_path.write_text("time,track_id,missing,snapshot\n")

    def update(self, frame, status):
        for tid, items in status.items():
            key = frozenset(items)
            if not key:
                self.logged.pop(tid, None)
                continue
            if self.logged.get(tid) == key:
                continue
            self.logged[tid] = key
            self._save(frame, tid, key)

    def _save(self, frame, tid, items):
        now = datetime.now()
        missing = "-".join(sorted(items))
        name = f"{now:%Y%m%d_%H%M%S}_id{tid}_{missing}.jpg"
        cv2.imwrite(str(self.folder / name), frame)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [now.isoformat(timespec="seconds"), tid, missing, name]
            )