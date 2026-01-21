import cv2
import numpy as np
from utils import clamp_box, iou
from config import (
    GOAL_CLASS_ID, GOAL_CONF_THRESH,
    RED_RATIO_THRESH, RED_S_MIN, RED_V_MIN,
    STICKY_USER_FRAMES, STICKY_IOU_MIN
)

def red_torso_ratio(frame_bgr, bbox) -> float:
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), h, w)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    bw = x2 - x1
    bh = y2 - y1

    # small padding to avoid arms/background
    pad_x = int(0.08 * bw)
    x1i = x1 + pad_x
    x2i = x2 - pad_x
    if x2i <= x1i:
        return 0.0

    # torso window: roughly chest to belly
    ty1 = y1 + int(0.18 * bh)
    ty2 = y1 + int(0.60 * bh)
    ty1 = max(0, min(h - 1, ty1))
    ty2 = max(0, min(h, ty2))
    if ty2 <= ty1:
        return 0.0

    roi = frame_bgr[ty1:ty2, x1i:x2i]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, RED_S_MIN, RED_V_MIN])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, RED_S_MIN, RED_V_MIN])
    upper2 = np.array([180, 255, 255])

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    red = cv2.bitwise_or(m1, m2)

    # light denoise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k, iterations=1)

    total = roi.shape[0] * roi.shape[1]
    return float(np.count_nonzero(red)) / max(1, total)

def find_bicycle_center(det_res):
    if det_res is None or det_res.boxes is None or len(det_res.boxes) == 0:
        return None

    xyxy = det_res.boxes.xyxy.cpu().numpy()
    conf = det_res.boxes.conf.cpu().numpy()
    cls = det_res.boxes.cls.cpu().numpy().astype(int)

    best_i = -1
    best_area = -1
    for i in range(len(xyxy)):
        if cls[i] != GOAL_CLASS_ID or conf[i] < GOAL_CONF_THRESH:
            continue
        x1, y1, x2, y2 = map(int, xyxy[i])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best_i = i

    if best_i < 0:
        return None

    x1, y1, x2, y2 = map(int, xyxy[best_i])
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def pick_user_red_then_sticky(det_res, frame_bgr, locked_box, lock_left):
    """
    Same behavior as your original:
    - If locked: pick person with best IoU (>= STICKY_IOU_MIN)
    - Else: acquire by red torso ratio >= RED_RATIO_THRESH
    """
    if det_res is None or det_res.boxes is None or len(det_res.boxes) == 0:
        return None, locked_box, max(0, lock_left - 1), 0.0

    boxes = det_res.boxes.xyxy.cpu().numpy()
    confs = det_res.boxes.conf.cpu().numpy()
    clss = det_res.boxes.cls.cpu().numpy().astype(int)

    # Follow lock by IoU
    if locked_box is not None and lock_left > 0:
        best_i, best_iou = -1, 0.0
        for i in range(len(boxes)):
            if clss[i] != 0:
                continue
            bb = tuple(map(int, boxes[i]))
            v = iou(locked_box, bb)
            if v > best_iou:
                best_iou, best_i = v, i

        if best_i >= 0 and best_iou >= STICKY_IOU_MIN:
            user_box = tuple(map(int, boxes[best_i]))
            red_score = red_torso_ratio(frame_bgr, user_box)
            return user_box, user_box, lock_left - 1, red_score

    # Acquire by red torso
    best_i, best_score, best_red = -1, -1.0, 0.0
    for i in range(len(boxes)):
        if clss[i] != 0:
            continue
        bb = tuple(map(int, boxes[i]))
        red_score = red_torso_ratio(frame_bgr, bb)
        if red_score < RED_RATIO_THRESH:
            continue

        score = float(confs[i]) + 0.4 * red_score
        if score > best_score:
            best_score = score
            best_i = i
            best_red = red_score

    if best_i >= 0:
        user_box = tuple(map(int, boxes[best_i]))
        return user_box, user_box, STICKY_USER_FRAMES, best_red

    return None, None, 0, 0.0
