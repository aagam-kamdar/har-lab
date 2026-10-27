"""
calibrate_seats.py - define one rectangle per seat and save seats.yaml

Usage:
    python calibrate_seats.py --source 0                 # webcam
    python calibrate_seats.py --source rtsp://...        # IP camera
    python calibrate_seats.py --source clip.mp4          # video file

Click TOP-LEFT then BOTTOM-RIGHT of each seat region (person + desk area).
Keys:  u = undo last seat   s/ESC = save and quit
Neighbors (left/right in IMAGE coordinates) are computed automatically.
"""
import argparse
import cv2
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="new lab video.mp4")
    ap.add_argument("--out", default="seats.yaml")
    ap.add_argument("--frame-idx", type=int, default=1000, help="Frame index to use for video calibration")
    a = ap.parse_args()

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not a.source.isdigit() and a.frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from source: {a.source}")

    rois, pts = [], []

    def redraw():
        img = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(rois, 1):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"D{i:02d}", (x1 + 6, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        for p in pts:
            cv2.circle(img, p, 4, (0, 0, 255), -1)
        cv2.imshow("seats", img)

    def click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            if len(pts) == 2:
                (xa, ya), (xb, yb) = pts
                rois.append([min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)])
                pts.clear()
            redraw()

    cv2.namedWindow("seats", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("seats", click)
    redraw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord("u") and rois:
            rois.pop()
            redraw()
        if k in (27, ord("s")):
            break
    cv2.destroyAllWindows()

    if not rois:
        print("No seats calibrated. Exiting without saving.")
        return

    # ---- auto neighbors: same row (centers within half a ROI height), nearest left/right
    seats = {}
    for i, r in enumerate(rois, 1):
        seats[f"D{i:02d}"] = {"roi": r, "neighbors": {"left": None, "right": None}}
    ids = list(seats)

    def cx(r): return (r[0] + r[2]) / 2
    def cy(r): return (r[1] + r[3]) / 2

    for sid in ids:
        r = seats[sid]["roi"]
        h = r[3] - r[1]
        row = [o for o in ids if o != sid and abs(cy(seats[o]["roi"]) - cy(r)) < 0.5 * h]
        left = [o for o in row if cx(seats[o]["roi"]) < cx(r)]
        right = [o for o in row if cx(seats[o]["roi"]) > cx(r)]
        if left:
            seats[sid]["neighbors"]["left"] = max(left, key=lambda o: cx(seats[o]["roi"]))
        if right:
            seats[sid]["neighbors"]["right"] = min(right, key=lambda o: cx(seats[o]["roi"]))

    with open(a.out, "w") as f:
        yaml.safe_dump({"seats": seats}, f, sort_keys=False)
    print(f"Saved {len(seats)} seats to {a.out}")


if __name__ == "__main__":
    main()
