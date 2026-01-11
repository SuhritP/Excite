import cv2
import numpy as np
import csv
import os

video_path = "C:/Users/sahas/OneDrive/Desktop/pupil/test.mp4"
output_csv = "circle_coordinates.csv"
output_video = "circle_coordinates_annotated.mp4"
threshold_value = 200


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print("FPS:", fps)
print("Frame count:", frame_count)
print("Duration:", frame_count / fps if fps else "unknown")

frame_num = 0
results = []  # (frame, x, y, validity)
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize writer once we know size/fps
    if out is None:
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1e-3 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # Threshold bright ring (adjust threshold_value if needed)
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Compute centroid
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        cx = int(xs.mean())
        cy = int(ys.mean())
        validity = 0
    else:
        cx, cy = -1, -1  # not found
        validity = 4

    # Overlay on frame
    if validity == 0:
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), 2)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    else:
        cv2.drawMarker(frame, (max(cx, 0), max(cy, 0)), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
    cv2.putText(frame, f"frame: {frame_num}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2)

    if out is not None:
        out.write(frame)

    results.append((frame_num, cx, cy, validity))
    frame_num += 1

cap.release()
if out is not None:
    out.release()

# Save CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "x", "y", "validity"])
    writer.writerows(results)

print(f"Saved {len(results)} frames to {output_csv}")
print(f"Saved annotated video to {output_video}")

