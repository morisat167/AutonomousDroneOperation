import time
import math
import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None

DET_MODEL_PATH = "yolo11m.pt"  

DET_CONF = 0.30
IMGSZ_DET = 640

# Flight
FLY = True
TAKEOFF_UP_CM = 80

MAX_LR = 40
MAX_FB = 35

Kp_lr = 0.12
Kd_lr = 0.05

Kp_fb = 220.0
Kd_fb = 80.0

DESIRED_AREA_RATIO = 0.10   # target bbox area / frame area
AREA_TOL = 0.03

DEADBAND_X = 18

# Lost behavior
LOST_TIMEOUT_SEC = 1.0

# Sticky user lock
STICKY_USER_FRAMES = 25
STICKY_IOU_MIN = 0.15

# Red shirt detection (torso-based)
RED_RATIO_THRESH = 0.20
RED_S_MIN = 80
RED_V_MIN = 60

# Video recording
SAVE_VIDEO = True
VIDEO_PATH = "follow_user_output.avi"
VIDEO_FPS = 20.0

# Display
W, H = 640, 480

def pick_device():
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))

def clamp_box(b, H, W):
    x1, y1, x2, y2 = b
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W - 1, int(x2)))
    y2 = max(0, min(H - 1, int(y2)))
    return (x1, y1, x2, y2)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def safe_start_stream(tello, warmup_ms=1000):
    tello.streamoff()
    time.sleep(0.2)
    tello.streamon()
    fr = tello.get_frame_read()
    start = time.time()
    frame = None
    while (time.time() - start) * 1000 < warmup_ms:
        frame = fr.frame
        if frame is not None and getattr(frame, "size", 0) != 0:
            break
        time.sleep(0.02)
    if frame is None or getattr(frame, "size", 0) == 0:
        raise RuntimeError("No frames from Tello. Check Wi-Fi/UDP.")
    return fr

def red_ratio_torso(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    Hh, Ww = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), Hh, Ww)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(0.08 * bw)
    x1i = x1 + pad_x
    x2i = x2 - pad_x
    if x2i <= x1i:
        return 0.0

    ty1 = y1 + int(0.18 * bh)
    ty2 = y1 + int(0.60 * bh)
    ty1 = max(0, min(Hh - 1, ty1))
    ty2 = max(0, min(Hh, ty2))
    if ty2 <= ty1:
        return 0.0

    roi = frame_bgr[ty1:ty2, x1i:x2i]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0,   RED_S_MIN, RED_V_MIN])
    upper_red1 = np.array([10,  255,       255])
    lower_red2 = np.array([170, RED_S_MIN, RED_V_MIN])
    upper_red2 = np.array([180, 255,       255])

    m1 = cv2.inRange(hsv, lower_red1, upper_red1)
    m2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red = cv2.bitwise_or(m1, m2)

    # small cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k, iterations=1)

    total = roi.shape[0] * roi.shape[1]
    red_pixels = int(np.count_nonzero(red))
    return red_pixels / max(1, total)

def pick_user_red_sticky(r_det, frame_bgr, sticky_bbox, sticky_count):
    """
    Returns: (user_bbox or None, new_sticky_bbox, new_sticky_count, rr)
    rr = torso red ratio (for debug)
    """
    if r_det is None or r_det.boxes is None or len(r_det.boxes) == 0:
        return None, sticky_bbox, max(0, sticky_count - 1), 0.0

    boxes = r_det.boxes.xyxy.cpu().numpy()
    confs = r_det.boxes.conf.cpu().numpy()
    clss  = r_det.boxes.cls.cpu().numpy().astype(int)

    # 1) keep sticky by IoU (no red test)
    if sticky_bbox is not None and sticky_count > 0:
        best_i = -1
        best_iou = 0.0
        for i in range(len(boxes)):
            if clss[i] != 0:  # PERSON
                continue
            bb = tuple(map(int, boxes[i]))
            v = iou_xyxy(sticky_bbox, bb)
            if v > best_iou:
                best_iou = v
                best_i = i
        if best_i >= 0 and best_iou >= STICKY_IOU_MIN:
            user_bbox = tuple(map(int, boxes[best_i]))
            rr = red_ratio_torso(frame_bgr, user_bbox)
            return user_bbox, user_bbox, sticky_count - 1, rr

    # 2) acquire by red torso
    best_i = -1
    best_score = -1.0
    best_rr = 0.0
    for i in range(len(boxes)):
        if clss[i] != 0:
            continue
        bb = tuple(map(int, boxes[i]))
        rr = red_ratio_torso(frame_bgr, bb)
        if rr >= RED_RATIO_THRESH:
            score = float(confs[i]) + 0.4 * rr
            if score > best_score:
                best_score = score
                best_i = i
                best_rr = rr

    if best_i >= 0:
        user_bbox = tuple(map(int, boxes[best_i]))
        return user_bbox, user_bbox, STICKY_USER_FRAMES, best_rr

    return None, None, 0, 0.0

def main():
    device = pick_device()
    print("[INFO] device:", device)

    det_model = YOLO(DET_MODEL_PATH)
    if device != "cpu":
        try:
            det_model.to(device)
        except Exception:
            pass
    try:
        det_model.fuse()
    except Exception:
        pass

    tello = Tello()
    print("[INFO] Connecting to Tello... (join Tello Wi-Fi)")
    try:
        tello.connect()
        print(f"[INFO] Battery: {tello.get_battery()}%")
    except Exception as e:
        print("[ERR] Tello connect:", e)
        return

    try:
        fr = safe_start_stream(tello, warmup_ms=1200)
    except Exception as e:
        print("[ERR]", e)
        try:
            tello.end()
        except Exception:
            pass
        return

    # Video writer
    video_writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (W, H))
        if not video_writer.isOpened():
            print("[WARN] Could not open VideoWriter. Disabling recording.")
            video_writer = None

    cv2.namedWindow("Follow View", cv2.WINDOW_NORMAL)

    # State
    sticky_bbox = None
    sticky_count = 0
    last_user_time = 0.0

    prev_time = time.time()
    prev_err_lr = 0.0
    prev_err_a  = 0.0

    flying = False

    try:
        if FLY:
            print("[INFO] Takeoff...")
            tello.takeoff()
            flying = True
            time.sleep(1.0)
            if TAKEOFF_UP_CM > 0:
                try:
                    tello.move_up(TAKEOFF_UP_CM)
                except Exception:
                    pass
                time.sleep(0.5)
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.3)

        while True:
            frame = fr.frame
            if frame is None or getattr(frame, "size", 0) == 0:
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            # Tello gives RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (W, H))
            display = frame.copy()

            # Timing
            now = time.time()
            dt = max(1e-3, now - prev_time)
            prev_time = now

            # DETECTION + TRACK
            dets = det_model.track(
                frame,
                conf=DET_CONF,
                imgsz=IMGSZ_DET,
                persist=True,
                verbose=False,
                device=device
            )
            r_det = dets[0] if dets else None

            # Pick user (red acquire + sticky)
            user_bbox, sticky_bbox, sticky_count, rr = pick_user_red_sticky(
                r_det, frame, sticky_bbox, sticky_count
            )

            user_pos = None
            area_ratio = 0.0

            if user_bbox is not None:
                x1, y1, x2, y2 = user_bbox
                user_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = max(1.0, float((x2 - x1) * (y2 - y1)))
                area_ratio = area / float(W * H)
                last_user_time = now

            # LOST?
            user_lost = (now - last_user_time) > LOST_TIMEOUT_SEC

            # ---- CONTROL ----
            lr = fb = 0
            if FLY and flying:
                if user_bbox is None or user_lost:
                    # hover (no user)
                    tello.send_rc_control(0, 0, 0, 0)
                    prev_err_lr = 0.0
                    prev_err_a  = 0.0
                else:
                    # center user with LR; keep distance with FB (area ratio)
                    cx = W // 2

                    ex = float(user_pos[0] - cx)               # + user right
                    dlr = (ex - prev_err_lr) / dt
                    prev_err_lr = ex

                    lr_cmd = Kp_lr * ex + Kd_lr * dlr
                    lr = clamp(lr_cmd, -MAX_LR, MAX_LR)
                    if abs(ex) < DEADBAND_X:
                        lr = 0

                    ea = float(DESIRED_AREA_RATIO - area_ratio)  # + too far -> forward
                    dea = (ea - prev_err_a) / dt
                    prev_err_a = ea

                    if abs(ea) < AREA_TOL:
                        fb_cmd = 0.0
                    else:
                        fb_cmd = Kp_fb * ea + Kd_fb * dea
                    fb = clamp(fb_cmd, -MAX_FB, MAX_FB)

                    tello.send_rc_control(lr, fb, 0, 0)

            # ---- DRAW ----
            if user_bbox is not None:
                x1, y1, x2, y2 = user_bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if user_pos is not None:
                    cv2.circle(display, user_pos, 6, (255, 0, 0), -1)

                # debug: red ratio + threshold
                color = (0, 255, 0) if rr >= RED_RATIO_THRESH else (0, 0, 255)
                cv2.putText(
                    display,
                    f"red_torso={rr:.2f} (th={RED_RATIO_THRESH:.2f})",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            if user_lost:
                cv2.putText(display, "USER LOST -> HOVER", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display, f"LR={lr} FB={fb} area={area_ratio:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            fps = 1.0 / max(1e-6, dt)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Record
            if video_writer is not None:
                video_writer.write(display)

            cv2.imshow("Follow View", display)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("l"):
                if FLY and flying:
                    try:
                        tello.land()
                    except Exception:
                        pass
                    flying = False
                break

    finally:
        print("[INFO] cleanup...")
        if FLY and flying:
            try:
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.2)
            except Exception:
                pass
            try:
                tello.land()
            except Exception:
                pass

        try:
            tello.streamoff()
        except Exception:
            pass
        try:
            tello.end()
        except Exception:
            pass
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
