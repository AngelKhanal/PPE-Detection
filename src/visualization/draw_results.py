import cv2

COLORS = {
    "helmet": (0, 200, 0),
    "vest": (0, 165, 255),
    "boots": (255, 100, 0),
}
HIDDEN = {"person"}  # drawn via tracks instead (draw_people)


def draw_detections(frame, detections):
    out = frame.copy()
    for d in detections:
        if d.label in HIDDEN:
            continue
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

def fit_to_screen(frame, max_w=1280, max_h=720):
    """Return a downscaled copy for display. Never enlarges."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)