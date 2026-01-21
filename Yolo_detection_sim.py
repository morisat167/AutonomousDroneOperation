import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

def red_ratio_in_roi(bgr_roi: np.ndarray) -> float:

    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    if total_pixels <= 0:
        return 0.0

    return red_pixels / float(total_pixels)


def run_yolo_on_video(
    video_path: str,
    model_path: str = "yolo11l.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 1024,
    show: bool = True,
    save: bool = False,
    out_path: str = "output_yolo.mp4",
    red_ratio_thresh: float = 0.08,   # >= 8% red pixels in shirt ROI -> "red shirt"
    draw_debug_shirt_roi: bool = False
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {out_path}")

    window_name = "YOLO + Red Shirt (press q to quit)"
    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )

        # Start with YOLO's normal plotted frame
        annotated = results[0].plot()

        # If detections exist, scan for persons (COCO class 0 = person)
        r0 = results[0]
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            confs = r0.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls_id, c in zip(xyxy, clss, confs):
                if cls_id != 0:  # only person
                    continue

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                # Shirt ROI: upper body (avoid head + legs)
                box_h = y2 - y1
                box_w = x2 - x1

                ys1 = y1 + int(0.20 * box_h)
                ys2 = y1 + int(0.65 * box_h)
                xs1 = x1 + int(0.15 * box_w)
                xs2 = x2 - int(0.15 * box_w)

                ys1 = max(y1, min(ys1, y2 - 1))
                ys2 = max(y1 + 1, min(ys2, y2))
                xs1 = max(x1, min(xs1, x2 - 1))
                xs2 = max(x1 + 1, min(xs2, x2))

                shirt_roi = frame[ys1:ys2, xs1:xs2]
                rr = red_ratio_in_roi(shirt_roi)

                is_red = rr >= red_ratio_thresh

                label = f"RED SHIRT {rr*100:.1f}%" if is_red else f"not red {rr*100:.1f}%"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255) if is_red else (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                if draw_debug_shirt_roi:
                    cv2.rectangle(
                        annotated,
                        (xs1, ys1),
                        (xs2, ys2),
                        (0, 0, 255) if is_red else (200, 200, 200),
                        2
                    )

        if show:
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        if save and writer is not None:
            writer.write(annotated)

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if save:
        print(f"Saved annotated video to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO on a video + detect red shirt on persons.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="yolo11x.pt", help="YOLO model path (e.g., yolov8n.pt, yolov8n-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--noshow", action="store_true", help="Disable live display window")
    parser.add_argument("--save", action="store_true", help="Save annotated output video")
    parser.add_argument("--out", default="output_yolo.mp4", help="Output video path (if --save)")
    parser.add_argument("--red_thresh", type=float, default=0.08, help="Red ratio threshold (0..1) for RED SHIRT")
    parser.add_argument("--debug_shirt_roi", action="store_true", help="Draw the shirt ROI rectangle")

    args = parser.parse_args()

    run_yolo_on_video(
        video_path=args.video,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show=not args.noshow,
        save=args.save,
        out_path=args.out,
        red_ratio_thresh=args.red_thresh,
        draw_debug_shirt_roi=args.debug_shirt_roi
    )
