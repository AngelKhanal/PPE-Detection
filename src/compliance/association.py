# Vertical band of the person box (0 = top, 1 = bottom) where each item's center must fall.
REGIONS = {
    "helmet": (0.0, 0.35),
    "vest": (0.15, 0.75),
    "boots": (0.80, 1.0),
}


def associate(tracks, detections):
    """Return {track_id: set of PPE labels found on that person}."""
    found = {t.id: set() for t in tracks}
    for d in detections:
        if d.label not in REGIONS:
            continue
        lo, hi = REGIONS[d.label]
        cx = (d.box[0] + d.box[2]) / 2
        cy = (d.box[1] + d.box[3]) / 2
        best = None  # (area, track_id); smallest containing person wins
        for t in tracks:
            x1, y1, x2, y2 = t.box
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            if not (lo <= (cy - y1) / (y2 - y1) <= hi):
                continue
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best[0]:
                best = (area, t.id)
        if best:
            found[best[1]].add(d.label)
    return found