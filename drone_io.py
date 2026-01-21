import time
from djitellopy import Tello

def safe_start_stream(tello: Tello, warmup_ms=1000):
    tello.streamoff()
    time.sleep(0.2)
    tello.streamon()

    frame_reader = tello.get_frame_read()
    start = time.time()

    while (time.time() - start) * 1000 < warmup_ms:
        frame = frame_reader.frame
        if frame is not None and getattr(frame, "size", 0) != 0:
            return frame_reader
        time.sleep(0.02)

    raise RuntimeError("No frames from Tello. Check Wi-Fi/UDP and stream state.")
