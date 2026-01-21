
import argparse
import random
import math
import cv2
import numpy as np
from ultralytics import YOLO

SEG_MODEL = "yolo11m-seg.pt"
DET_MODEL = "yolov11m.pt"

IMGSZ = 768
SEG_CONF = 0.20
DET_CONF = 0.25


PERSON = 0
CAR = 2
BUS = 5
TRUCK = 7
VEHICLE_CLASSES = {CAR, BUS, TRUCK}

# RRT
RRT_MAX_ITER = 4000
RRT_STEP = 25
GOAL_BIAS = 0.20
GOAL_REACHED_TOL = 30

OBSTACLE_DILATE = 7


ANIM_FPS = 20
ANIM_REPEAT_FRAMES_PER_POINT = 2  # higher = slower movement


class Node:
    __slots__ = ("x", "y", "parent")
    def __init__(self, x, y, parent=None):
        self.x = int(x)
        self.y = int(y)
        self.parent = parent


def is_free(mask_free, x, y):
    h, w = mask_free.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return mask_free[y, x] != 0


def segment_free_line(mask_free, x1, y1, x2, y2, step=2):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return is_free(mask_free, x1, y1)
    n = max(1, int(dist / step))
    for i in range(n + 1):
        t = i / n
        x = int(round(x1 + t * dx))
        y = int(round(y1 + t * dy))
        if not is_free(mask_free, x, y):
            return False
    return True


def nearest(nodes, x, y):
    best = None
    best_d = 1e18
    for nd in nodes:
        d = (nd.x - x) ** 2 + (nd.y - y) ** 2
        if d < best_d:
            best_d = d
            best = nd
    return best


def steer(x1, y1, x2, y2, step):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist <= step:
        return int(x2), int(y2)
    ux = dx / dist
    uy = dy / dist
    return int(round(x1 + ux * step)), int(round(y1 + uy * step))


def rrt_plan(mask_free, start, goal):
    h, w = mask_free.shape[:2]
    sx, sy = start
    gx, gy = goal

    if not is_free(mask_free, sx, sy) or not is_free(mask_free, gx, gy):
        return None

    nodes = [Node(sx, sy, None)]

    for _ in range(RRT_MAX_ITER):
        if random.random() < GOAL_BIAS:
            rx, ry = gx, gy
        else:
            rx = random.randint(0, w - 1)
            ry = random.randint(0, h - 1)

        nn = nearest(nodes, rx, ry)
        nx, ny = steer(nn.x, nn.y, rx, ry, RRT_STEP)

        if not is_free(mask_free, nx, ny):
            continue
        if not segment_free_line(mask_free, nn.x, nn.y, nx, ny):
            continue

        new_node = Node(nx, ny, nn)
        nodes.append(new_node)

        if (nx - gx) ** 2 + (ny - gy) ** 2 <= GOAL_REACHED_TOL ** 2:
            if segment_free_line(mask_free, nx, ny, gx, gy):
                goal_node = Node(gx, gy, new_node)
                path = []
                cur = goal_node
                while cur is not None:
                    path.append((cur.x, cur.y))
                    cur = cur.parent
                path.reverse()
                return path

    return None

def center_of_xyxy(xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_label(img, text, xy, scale=0.65, thickness=2):
    x, y = int(xy[0]), int(xy[1])
    y = max(20, y)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)


def point_in_box(px, py, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def make_mask_preview(free):
    return cv2.cvtColor(free, cv2.COLOR_GRAY2BGR)


CLICK_GOAL = None         # (x,y) chosen by click
CLICK_GOAL_VEHICLE = None # xyxy bbox if clicked inside vehicle, else None
REQUEST_REPLAN = False


def main():
    global CLICK_GOAL, CLICK_GOAL_VEHICLE, REQUEST_REPLAN

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_img", default="rrt_click_goal_output.png")
    ap.add_argument("--out_vid", default="rrt_click_goal_sim.mp4")
    ap.add_argument("--nosavevid", action="store_true", help="Don't save video")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    H, W = img.shape[:2]

    seg = YOLO(SEG_MODEL)
    det = YOLO(DET_MODEL)

    seg_res = seg.predict(img, imgsz=IMGSZ, conf=SEG_CONF, verbose=False)[0]

    persons = []   # (conf, xyxy)
    vehicles = []  # (conf, cls, xyxy)

    seg_boxes = None
    seg_clss = None
    seg_confs = None
    seg_masks = None

    if seg_res.boxes is not None and len(seg_res.boxes) > 0:
        seg_boxes = seg_res.boxes.xyxy.cpu().numpy()
        seg_clss = seg_res.boxes.cls.cpu().numpy().astype(int)
        seg_confs = seg_res.boxes.conf.cpu().numpy()
        seg_masks = seg_res.masks.data.cpu().numpy() if seg_res.masks is not None else None

        for i in range(len(seg_boxes)):
            cls = int(seg_clss[i])
            conf = float(seg_confs[i])
            xyxy = seg_boxes[i]
            if cls == PERSON:
                persons.append((conf, xyxy))
            if cls in VEHICLE_CLASSES:
                vehicles.append((conf, cls, xyxy))


    if len(persons) == 0 or len(vehicles) == 0:
        det_res = det.predict(img, imgsz=IMGSZ, conf=DET_CONF, verbose=False)[0]
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            dboxes = det_res.boxes.xyxy.cpu().numpy()
            dclss = det_res.boxes.cls.cpu().numpy().astype(int)
            dconfs = det_res.boxes.conf.cpu().numpy()
            for i in range(len(dboxes)):
                cls = int(dclss[i])
                conf = float(dconfs[i])
                xyxy = dboxes[i]
                if cls == PERSON:
                    persons.append((conf, xyxy))
                if cls in VEHICLE_CLASSES:
                    vehicles.append((conf, cls, xyxy))

    if len(persons) == 0:
        print("No person detected.")
        cv2.imshow("RRT Simulation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    persons.sort(key=lambda x: x[0], reverse=True)
    user_conf, user_xyxy = persons[0]
    user_center = center_of_xyxy(user_xyxy)

    print(f"Detected persons: {len(persons)} | vehicles: {len(vehicles)}")
    print("Click on a car (or anywhere) in the 'RRT Simulation' window to set the goal.")
    print("Keys: r=replan, q/ESC=quit")

    def on_mouse(event, x, y, flags, param):
        global CLICK_GOAL, CLICK_GOAL_VEHICLE, REQUEST_REPLAN
        if event == cv2.EVENT_LBUTTONDOWN:
            # If click inside a vehicle bbox -> snap goal to that vehicle center
            chosen = None
            for (conf, cls, xyxy) in vehicles:
                if point_in_box(x, y, xyxy):
                    chosen = xyxy
                    break

            if chosen is not None:
                CLICK_GOAL_VEHICLE = chosen
                CLICK_GOAL = center_of_xyxy(chosen)
                print(f"[GOAL] Clicked vehicle -> goal snapped to vehicle center: {CLICK_GOAL}")
            else:
                CLICK_GOAL_VEHICLE = None
                CLICK_GOAL = (x, y)
                print(f"goal found")

            REQUEST_REPLAN = True

    cv2.namedWindow("RRT Simulation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("RRT Simulation", on_mouse)

    cv2.namedWindow("Segmentation Mask", cv2.WINDOW_NORMAL)

    CLICK_GOAL = (W // 2, H // 4)
    CLICK_GOAL_VEHICLE = None
    REQUEST_REPLAN = True

    video = None

    while True:
        goal_center = CLICK_GOAL

        obstacle = np.zeros((H, W), dtype=np.uint8)

        if seg_masks is not None and seg_boxes is not None:
            for i in range(len(seg_boxes)):
                cls = int(seg_clss[i])
                xyxy = seg_boxes[i]

                if cls not in VEHICLE_CLASSES:
                    continue

                # If goal is a chosen vehicle bbox, exclude it from obstacles
                if CLICK_GOAL_VEHICLE is not None:
                    # exclude the vehicle you clicked
                    if point_in_box(goal_center[0], goal_center[1], xyxy):
                        # (goal center lies in it)
                        continue

                m = (seg_masks[i] > 0.5).astype(np.uint8) * 255
                if m.shape[0] != H or m.shape[1] != W:
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                obstacle = cv2.bitwise_or(obstacle, m)
        else:
            for (conf, cls, xyxy) in vehicles:
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                # if clicked inside this bbox and we treat it as goal, skip it
                if CLICK_GOAL_VEHICLE is not None and point_in_box(goal_center[0], goal_center[1], xyxy):
                    continue
                cv2.rectangle(obstacle, (x1, y1), (x2, y2), 255, -1)

        # dilate obstacles
        if OBSTACLE_DILATE > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OBSTACLE_DILATE, OBSTACLE_DILATE))
            obstacle = cv2.dilate(obstacle, k, iterations=1)

        free = cv2.bitwise_not(obstacle)

        # make sure start/goal are free pixels
        cv2.circle(free, user_center, 10, 255, -1)
        cv2.circle(free, goal_center, 10, 255, -1)

        # Show mask window
        mask_preview = make_mask_preview(free)
        # draw start/goal on mask preview
        cv2.circle(mask_preview, user_center, 6, (0, 255, 0), -1)
        cv2.circle(mask_preview, goal_center, 6, (255, 0, 0), -1)
        cv2.imshow("Segmentation Mask", mask_preview)

        # If goal changed or replan requested, compute path
        path = None
        if REQUEST_REPLAN:
            REQUEST_REPLAN = False
            path = rrt_plan(free, user_center, goal_center)
            if path is None:
                print("[RRT] No path found.")

            # (Re)initialize video when planning succeeds and saving enabled
            if (not args.nosavevid) and path is not None:
                if video is not None:
                    video.release()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video = cv2.VideoWriter(args.out_vid, fourcc, ANIM_FPS, (W, H))
                if not video.isOpened():
                    print("[WARN] VideoWriter failed. Try changing output to .avi")
                    video = None

        base = img.copy()


        ux1, uy1, ux2, uy2 = [int(v) for v in user_xyxy]
        cv2.rectangle(base, (ux1, uy1), (ux2, uy2), (0, 255, 0), 2)
        draw_label(base, f"USER person conf={user_conf:.2f}", (ux1, uy1 - 8))
        cv2.circle(base, user_center, 7, (0, 255, 0), -1)

        cv2.circle(base, goal_center, 8, (255, 0, 0), -1)
        draw_label(base, "GOAL", (goal_center[0] + 10, goal_center[1]))

        if "LAST_PATH" not in locals():
            LAST_PATH = None
        if path is not None:
            LAST_PATH = path

        # draw path
        if LAST_PATH is not None:
            for i in range(len(LAST_PATH) - 1):
                cv2.line(base, LAST_PATH[i], LAST_PATH[i + 1], (0, 255, 255), 3)
            draw_label(base, f"PATH points={len(LAST_PATH)}", (20, 40))
        else:
            draw_label(base, "NO PATH (click another goal or press r)", (20, 40))

        cv2.imshow("RRT Simulation", base)
        if LAST_PATH is not None:
            if "anim_idx" not in locals():
                anim_idx = 0
            for _ in range(ANIM_REPEAT_FRAMES_PER_POINT):
                frame = base.copy()
                px, py = LAST_PATH[min(anim_idx, len(LAST_PATH) - 1)]
                cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)
                draw_label(frame, "USER (moving)", (px + 12, py))
                cv2.imshow("RRT Simulation", frame)

                if video is not None:
                    video.write(frame)

                k = cv2.waitKey(int(1000 / ANIM_FPS)) & 0xFF
                if k == ord('r'):
                    REQUEST_REPLAN = True
                    anim_idx = 0
                    LAST_PATH = None
                    break
                if k == 27 or k == ord('q'):
                    if video is not None:
                        video.release()
                    cv2.destroyAllWindows()
                    return

            anim_idx += 1
            if anim_idx >= len(LAST_PATH):
                anim_idx = 0  # loop movement
        else:
            # no path: just wait for input
            k = cv2.waitKey(30) & 0xFF
            if k == ord('r'):
                REQUEST_REPLAN = True
            if k == 27 or k == ord('q'):
                if video is not None:
                    video.release()
                cv2.destroyAllWindows()
                return


if __name__ == "__main__":
    main()
