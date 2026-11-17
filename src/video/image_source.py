import cv2


def load_image(path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return frame