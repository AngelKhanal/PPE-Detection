import os
import re
import threading
import time

import cv2


def _mask(source):
    """Hide user:password in stream URLs before printing."""
    return re.sub(r"://[^@/]*@", "://***@", str(source))


def camera_frames(source=0, reconnect_delay=3.0, max_retries=5):
    """Yield the latest frame from a webcam index or stream URL (RTSP/HTTP).
    A reader thread drops old frames so processing never lags behind live.
    Reconnects if the stream drops."""
    if isinstance(source, str) and source.lower().startswith("rtsp"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    retries = 0
    while retries <= max_retries:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            retries += 1
            print(f"Cannot open {_mask(source)}, retry {retries}/{max_retries}")
            time.sleep(reconnect_delay)
            continue

        lock = threading.Lock()
        stop = threading.Event()
        state = {"frame": None, "alive": True}

        def grab():
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    with lock:
                        state["alive"] = False
                    return
                with lock:
                    state["frame"] = frame

        thread = threading.Thread(target=grab, daemon=True)
        thread.start()
        try:
            while True:
                with lock:
                    frame, alive = state["frame"], state["alive"]
                    state["frame"] = None  # each frame processed once
                if frame is not None:
                    retries = 0
                    yield frame
                elif not alive:
                    break
                else:
                    time.sleep(0.005)
        finally:
            stop.set()
            thread.join(timeout=2)
            cap.release()

        retries += 1
        print(f"Stream lost, reconnecting {retries}/{max_retries}")
        time.sleep(reconnect_delay)
    raise RuntimeError(f"Could not read from {_mask(source)}")