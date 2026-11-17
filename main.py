import argparse
from pathlib import Path

import cv2
import yaml

from src.compliance.association import associate
from src.compliance.evidence import EvidenceLogger
from src.compliance.rules import ComplianceRules
from src.detection.detector import Detector
from src.tracking.tracker import PersonTracker
from src.video.camera_source import camera_frames
from src.video.image_source import load_image
from src.video.video_source import video_fps, video_frames
from src.visualization.draw_results import draw_detections, draw_people

CCTV_CONFIG = Path("configs/cctv.local.yaml")


def load_config():
    return yaml.safe_load(Path("configs/model.yaml").read_text())


def load_cctv_url():
    if not CCTV_CONFIG.exists():
        raise SystemExit(
            f"{CCTV_CONFIG} not found. Copy configs/cctv.example.yaml to it and set url."
        )
    url = yaml.safe_load(CCTV_CONFIG.read_text()).get("url")
    if not url or "CAMERA_IP" in url:
        raise SystemExit(f"Set a real url in {CCTV_CONFIG}")
    return url


def run_image(path, out):
    cfg = load_config()
    frame = load_image(path)
    detections = Detector().detect(frame)
    people = PersonTracker(**cfg["person"]).detect_once(frame)
    rules = ComplianceRules(cfg["required"], 1)  # one image = one frame
    found = associate(people, detections)
    status = rules.update(found)

    for p in people:
        missing = sorted(status[p.id])
        wearing = sorted(found[p.id])
        print(f"person {p.id}: wearing {wearing or 'nothing'}, "
              f"{'MISSING ' + ', '.join(missing) if missing else 'OK'}")
    if not people:
        print("no people detected")

    annotated = draw_people(draw_detections(frame, detections), people, status)
    EvidenceLogger().update(annotated, status)

    out_path = Path(out or f"outputs/images/{Path(path).stem}_out.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved {out_path}")


def run_stream(frames, save_path=None, fps=30.0):
    cfg = load_config()
    detector = Detector()
    tracker = PersonTracker(**cfg["person"])
    rules = ComplianceRules(cfg["required"], cfg["violation_frames"])
    evidence = EvidenceLogger()

    writer = None
    try:
        for frame in frames:
            detections = detector.detect(frame)
            tracks = tracker.update(frame)
            status = rules.update(associate(tracks, detections))
            out = draw_people(draw_detections(frame, detections), tracks, status)
            evidence.update(out, status)

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
    group.add_argument("--stream", metavar="URL", help="RTSP/HTTP stream URL")
    group.add_argument("--cctv", action="store_true",
                       help="use the stream URL from configs/cctv.local.yaml")
    parser.add_argument("--out", default=None, help="output path for image or video")
    args = parser.parse_args()

    if args.image:
        run_image(args.image, args.out)
    elif args.video:
        out = Path(args.out or f"outputs/videos/{Path(args.video).stem}_out.mp4")
        run_stream(video_frames(args.video), out, video_fps(args.video))
    elif args.cctv:
        run_stream(camera_frames(load_cctv_url()))
    elif args.stream:
        run_stream(camera_frames(args.stream))
    else:
        run_stream(camera_frames(args.camera))


if __name__ == "__main__":
    main()