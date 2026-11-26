import time
import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
from hns_projekt2.train.config import choose_device
from hns_projekt2.train.models import build_model
from hns_projekt2.train.train import MODELS_KEYS, train_model

COLORS = {"bad_wear": (0, 0, 255), "good_wear": (0, 200, 0)}   # BGR: red / green

def square_crop(x1, y1, x2, y2, pad: float, w: int, h: int):
    """Square box around the face, enlarged by `pad`, clipped to the frame.
    Training images are square, full faces with some margin. A tight, tall detector
    box would get center-cropped by the transform and could lose the chin."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + pad) / 2
    return (int(max(0, cx - half)), int(max(0, cy - half)),
            int(min(w, cx + half)), int(min(h, cy + half)))

def draw_label(frame, box, text: str, color):
    x1, y1, x2, y2 = box
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    top = max(y1, th + base + 4)   # keep the label inside the frame at the top edge
    cv2.rectangle(frame, (x1, top - th - base - 4), (x1 + tw + 4, top), color, -1)
    cv2.putText(frame, text, (x1 + 2, top - base - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Face detection + mask-wearing classification on video")
    parser.add_argument("-d", "--detector", type=str, required=True, help="YOLO face detector weights")
    parser.add_argument("-c", "--classifier", type=str, required=True, help="Trained classifier")
    parser.add_argument("-s", "--source", type=str, default="0", help="camera index, video file or stream URL")
    parser.add_argument("--conf", type=float, default=0.5, help="face detection confidence threshold")
    parser.add_argument("--pad", type=float, default=0.2, help="extra margin around each face crop")
    parser.add_argument("--min_size", type=int, default=24, help="skip faces smaller than this (px)")
    parser.add_argument("--save", type=str, default=None, help="optional output video path, e.g. out.mp4")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose messaging from ultralytics models")
    args = parser.parse_args()

    device = choose_device()
    detector = YOLO(Path(args.detector))
    classifier = YOLO(Path(args.classifier))

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    if args.save:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30   # webcams often report 0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, size)

    prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # 1. detect faces
        det = detector(frame, conf=args.conf, device=device, verbose=args.verbose)[0]

        # 2. cut them out
        crops, boxes = [], []
        for x1, y1, x2, y2 in det.boxes.xyxy.int().tolist():
            a, b, c, d = square_crop(x1, y1, x2, y2, args.pad, w, h)
            if c - a < args.min_size or d - b < args.min_size:
                continue
            crops.append(frame[b:d, a:c])
            boxes.append((x1, y1, x2, y2))

        # 3. classify all faces of this frame in one batch, 4. draw
        if crops:
            results = classifier.predict(crops, device=device, verbose=args.verbose)
            for box, r in zip(boxes, results):
                label = r.names[r.probs.top1]          # class name stored in the checkpoint
                conf = float(r.probs.top1conf)         # probability of that class
                draw_label(frame, box, f"{label} {conf:.2f}", COLORS.get(label, (255, 255, 0)))

        now = time.perf_counter()
        cv2.putText(frame, f"{1 / (now - prev):.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        prev = now

        if writer is not None:
            writer.write(frame)
        cv2.imshow("Mask detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
