import argparse
from pathlib import Path

import cv2
import yaml

from src.compliance.association import associate
from src.compliance.evidence import EvidenceLogger  # added
from src.compliance.rules import ComplianceRules
from src.detection.detector import Detector
from src.tracking.tracker import PersonTracker
from src.video.camera_source import camera_frames
from src.video.image_source import load_image
from src.video.video_source import video_fps, video_frames
from src.visualization.draw_results import draw_detections, draw_people


def run_image(path, out):
    frame = load_image(path)
    detections = Detector().detect(frame)
    for d in detections:
        x1, y1, x2, y2 = d.box
        print(f"{d.label:8s} {d.conf:.2f} ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    out_path = Path(out or f"outputs/images/{Path(path).stem}_out.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), draw_detections(frame, detections))
    print(f"Saved {out_path}")


def run_stream(frames, save_path=None, fps=30.0):
    cfg = yaml.safe_load(Path("configs/model.yaml").read_text())
    detector = Detector()
    tracker = PersonTracker(**cfg["person"])
    rules = ComplianceRules(cfg["required"], cfg["violation_frames"])
    evidence = EvidenceLogger()  # added

    writer = None
    try:
        for frame in frames:
            detections = detector.detect(frame)
            tracks = tracker.update(frame)
            status = rules.update(associate(tracks, detections))
            out = draw_people(draw_detections(frame, detections), tracks, status)
            evidence.update(out, status)  # added

            if save_path:
                if writer is None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(
                        str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                writer.write(out)

            cv2.imshow("ppe", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if writer:
            writer.release()
            print(f"Saved {save_path}")
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image")
    group.add_argument("--video")
    group.add_argument("--camera", type=int, metavar="INDEX")
    parser.add_argument("--out", default=None, help="output path for image or video")
    args = parser.parse_args()

    if args.image:
        run_image(args.image, args.out)
    elif args.video:
        out = Path(args.out or f"outputs/videos/{Path(args.video).stem}_out.mp4")
        run_stream(video_frames(args.video), out, video_fps(args.video))
    else:
        run_stream(camera_frames(args.camera))


if __name__ == "__main__":
    main()