import cv2
from ultralytics import YOLO

# Small pretrained local YOLO model
model = YOLO("yolo26n.pt")

video_path = "test_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

while True:
    success, frame = cap.read()

    if not success:
        break

    # ByteTrack + persistent tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],      # COCO class 0 = person
        conf=0.3,
        verbose=False
    )

    result = results[0]

    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"ID {track_id} {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            print(
                f"track_id={track_id} "
                f"conf={conf:.2f} "
                f"bbox=({x1},{y1},{x2},{y2})"
            )

    cv2.imshow("Local ByteTrack Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()