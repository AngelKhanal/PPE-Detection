import cv2

COLORS = {
    "helmet": (0, 200, 0),
    "vest": (0, 165, 255),
    "boots": (255, 100, 0),
    "unknown": (150, 150, 150),
}


def draw_detections(frame, detections):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d.box)
        color = COLORS.get(d.label, (0, 0, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{d.label} {d.conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def draw_people(frame, tracks, status):
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        missing = status.get(t.id, set())
        color = (0, 0, 255) if missing else (0, 200, 0)
        text = f"ID {t.id} " + (f"MISSING: {', '.join(sorted(missing))}" if missing else "OK")
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, text, (x1, min(y2 + 18, out.shape[0] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out