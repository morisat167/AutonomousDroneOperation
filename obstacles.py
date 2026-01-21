import cv2
import numpy as np
from config import (
    GROUND_TOP_Y_RATIO, GROUND_TOP_WIDTH_RATIO,
    GROUND_DILATE, GROUND_ERODE
)

def build_ground_mask(h, w, user_box=None, mode="fixed"):
    ground = np.zeros((h, w), dtype=np.uint8)

    if mode == "dynamic" and user_box is not None:
        x1, y1, x2, y2 = user_box
        top_y = int(max(0, min(h - 1, y2 - 0.35 * (y2 - y1))))
        top_w = int(w * 0.50)
        cx = (x1 + x2) // 2
        left_top = max(0, cx - top_w // 2)
        right_top = min(w - 1, cx + top_w // 2)
        pts = np.array(
            [[left_top, top_y], [right_top, top_y], [w - 1, h - 1], [0, h - 1]],
            dtype=np.int32
        )
    else:
        top_y = int(h * GROUND_TOP_Y_RATIO)
        top_w = int(w * GROUND_TOP_WIDTH_RATIO)
        left_top = (w - top_w) // 2
        right_top = left_top + top_w
        pts = np.array(
            [[left_top, top_y], [right_top, top_y], [w - 1, h - 1], [0, h - 1]],
            dtype=np.int32
        )

    cv2.fillPoly(ground, [pts], 255)

    if GROUND_DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROUND_DILATE, GROUND_DILATE))
        ground = cv2.dilate(ground, k, iterations=1)

    if GROUND_ERODE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROUND_ERODE, GROUND_ERODE))
        ground = cv2.erode(ground, k, iterations=1)

    return ground

def build_obstacle_mask(seg_res, shape_hw, exclude_classes):
    h, w = shape_hw
    mask_view = np.full((h, w, 3), 255, dtype=np.uint8)
    nav_mask = np.full((h, w), 255, dtype=np.uint8)

    if seg_res is None:
        return mask_view, nav_mask

    if not (hasattr(seg_res, "masks") and seg_res.masks is not None and seg_res.boxes is not None):
        return mask_view, nav_mask

    classes = seg_res.boxes.cls.cpu().numpy().astype(int)

    if seg_res.masks.data is None:
        return mask_view, nav_mask

    masks = seg_res.masks.data.cpu().numpy()
    for i, m in enumerate(masks):
        cls_id = classes[i] if i < len(classes) else -1
        if cls_id in exclude_classes:
            continue

        m = (m > 0.5).astype(np.uint8)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_view[m.astype(bool)] = 0
        nav_mask[m.astype(bool)] = 0

    return mask_view, nav_mask
