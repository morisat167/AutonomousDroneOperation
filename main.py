import time
import cv2
import torch
from ultralytics import YOLO
from djitellopy import Tello

from config import *
from utils import distance, put_status
from drone_io import safe_start_stream
from vision import pick_user_red_then_sticky, find_bicycle_center
from obstacles import build_obstacle_mask, build_ground_mask
from rrt import (
    run_rrt, build_path,
    closest_path_idx, user_off_path, path_ok_from
)
from control import FollowController


def choose_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main():
    device = choose_device()
    print("[info] device:", device)

    seg_model = YOLO(SEG_MODEL_PATH)
    det_model = YOLO(DET_MODEL_PATH)

    if device != "cpu":
        seg_model.to(device)
        det_model.to(device)

    seg_model.fuse()
    det_model.fuse()

    tello = Tello()
    print("[info] connecting to tello (connect to tello wi-fi)...")
    tello.connect()
    print(f"[info] battery: {tello.get_battery()}%")

    frame_reader = safe_start_stream(tello, warmup_ms=1200)

    w, h = 640, 480
    controller = FollowController(
        w, h,
        MAX_LR, MAX_FB,
        KP_FB, KD_FB, DESIRED_AREA_RATIO, AREA_TOL,
        KP_LR, KD_LR, DEADBAND_X
    )

    video_writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (w, h))
        if not video_writer.isOpened():
            print("[warn] videowriter couldn't open -> recording disabled")
            video_writer = None

    cv2.namedWindow("Mission View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Obstacle Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Ground Mask", cv2.WINDOW_NORMAL)

    path = []
    goal_xy = None
    goal_reached = False

    locked_user_box = None
    lock_left = 0

    prev_time = time.time()

    flying_now = False
    goal_last_seen = 0.0

    try:
        if FLY:
            print("[info] takeoff...")
            tello.takeoff()
            flying_now = True
            time.sleep(1.0)

            if GO_UP_CM > 0:
                try:
                    tello.move_up(GO_UP_CM)
                except Exception as e:
                    print("[warn] move_up failed:", e)
                time.sleep(0.5)

            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.3)

        while True:
            raw = frame_reader.frame
            if raw is None or getattr(raw, "size", 0) == 0:
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (w, h))
            display = frame.copy()

            # --- Detection / Tracking ---
            det_list = det_model.track(
                frame,
                conf=DET_CONF,
                imgsz=IMGSZ_DET,
                persist=True,
                verbose=False,
                device=device
            )
            det_res = det_list[0] if det_list else None

            user_box, locked_user_box, lock_left, red_score = pick_user_red_then_sticky(
                det_res, frame, locked_user_box, lock_left
            )

            user_xy = None
            area_ratio = 0.0
            if user_box is not None:
                x1, y1, x2, y2 = user_box
                user_xy = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = max(1.0, float((x2 - x1) * (y2 - y1)))
                area_ratio = area / float(w * h)

            # --- Goal (bicycle) ---
            g = find_bicycle_center(det_res)
            if g is not None:
                goal_xy = g
                goal_last_seen = time.time()

            # --- Segmentation obstacles ---
            seg_list = seg_model(
                frame,
                conf=SEG_CONF,
                imgsz=IMGSZ_SEG,
                verbose=False,
                device=device
            )
            seg_res = seg_list[0] if seg_list else None
            obstacle_view, nav_mask = build_obstacle_mask(seg_res, (h, w), {0, GOAL_CLASS_ID})

            # --- Ground ROI ---
            ground_mask = build_ground_mask(h, w, user_box, GROUND_MODE)
            nav_mask = cv2.bitwise_and(nav_mask, ground_mask)
            obstacle_view[ground_mask == 0] = (0, 0, 0)

            # --- Goal reached ---
            if (not goal_reached) and (user_xy is not None) and (goal_xy is not None):
                if distance(user_xy, goal_xy) <= GOAL_REACH_DIST_PX:
                    goal_reached = True
                    path = []

            # --- Plan / replan (RRT) ---
            if (not goal_reached) and (user_xy is not None) and (goal_xy is not None):
                start_idx = closest_path_idx(user_xy, path) if path else 0

                need_replan = False
                if (not path) or user_off_path(user_xy, path, PATH_DEVIATION_TOL):
                    need_replan = True
                elif not path_ok_from(path, start_idx, nav_mask):
                    need_replan = True

                if need_replan:
                    goal_node = run_rrt(
                        user_xy, goal_xy, nav_mask,
                        max_iter=RRT_MAX_ITER,
                        step_size=RRT_STEP,
                        goal_tol=RRT_GOAL_TOL,
                        bias=GOAL_BIAS
                    )
                    path = build_path(goal_node)

            # --- Timing ---
            now = time.time()
            dt = max(1e-3, now - prev_time)
            prev_time = now

            # --- Flight control ---
            if FLY and flying_now:
                goal_missing = (goal_xy is None) or ((now - goal_last_seen) > GOAL_LOST_SEC)

                if SEARCH_FOR_GOAL and goal_missing:
                    tello.send_rc_control(0, 0, 0, 25)
                    put_status(display, "Searching for bicycle...", 30, (0, 0, 255))

                elif user_xy is None:
                    tello.send_rc_control(0, 0, 0, 0)
                    put_status(display, "Goal found. Waiting for red shirt person...", 30, (0, 165, 255))
                    controller.reset()

                else:
                    lr, fb = controller.compute(user_xy, area_ratio, dt)
                    tello.send_rc_control(lr, fb, 0, 0)

            # --- Draw goal ---
            if goal_xy is not None:
                cv2.circle(display, goal_xy, 10, (0, 0, 255), -1)
                cv2.putText(display, GOAL_LABEL, (goal_xy[0] + 15, goal_xy[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- Draw user ---
            if user_box is not None and user_xy is not None:
                x1, y1, x2, y2 = user_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(display, user_xy, 6, (255, 0, 0), -1)
                cv2.putText(display,
                            f"USER red={red_score:.2f}/(th={RED_RATIO_THRESH:.2f})",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if goal_reached:
                cv2.putText(display, "GOAL REACHED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # --- Draw remaining path only ---
            if (not goal_reached) and path and user_xy:
                start_idx = closest_path_idx(user_xy, path)
                for i in range(start_idx, len(path) - 1):
                    cv2.line(display, path[i], path[i + 1], (255, 255, 0), 3)

            fps = 1.0 / max(1e-6, dt)
            cv2.putText(display, f"fps: {fps:.1f}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if video_writer is not None:
                video_writer.write(display)

            cv2.imshow("Mission View", display)
            cv2.imshow("Obstacle Mask", obstacle_view)
            cv2.imshow("Binary Mask", cv2.cvtColor(nav_mask, cv2.COLOR_GRAY2BGR))
            cv2.imshow("Ground Mask", cv2.cvtColor(ground_mask, cv2.COLOR_GRAY2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                if FLY and flying_now:
                    try:
                        tello.land()
                    except Exception as e:
                        print("[warn] land failed:", e)
                    flying_now = False
                break

    finally:
        print("[info] cleanup...")
        if FLY and flying_now:
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.2)
            tello.land()

        tello.streamoff()
        tello.end()

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
