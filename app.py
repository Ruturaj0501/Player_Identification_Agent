import cv2
from ultralytics import YOLO
import numpy as np


video_path = "15sec_input_720p.mp4"
model_path = "best.pt"
output_path = "reid_output.avi"

model = YOLO(model_path)

next_id = 0
tracked_players = {}
disappeared = {}
max_disappeared = 30
max_dist = 50

def get_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

cap = cv2.VideoCapture(video_path)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.5 or cls != 0:  # 0 = player
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append(((x1, y1, x2, y2), get_center(x1, y1, x2, y2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    updated = {}
    used = set()

    for box, center in detections:
        matched = False
        for pid, old_center in tracked_players.items():
            dist = np.linalg.norm(np.array(center) - np.array(old_center))
            if dist < max_dist and pid not in used:
                updated[pid] = center
                disappeared[pid] = 0
                used.add(pid)
                matched = True
                break
        if not matched:
            updated[next_id] = center
            disappeared[next_id] = 0
            next_id += 1

    for pid in list(tracked_players):
        if pid not in updated:
            disappeared[pid] += 1
            if disappeared[pid] > max_disappeared:
                del tracked_players[pid]
                del disappeared[pid]

    tracked_players.update(updated)

    for pid, center in tracked_players.items():
        cv2.circle(frame, center, 4, (255, 0, 0), -1)
        cv2.putText(frame, f"ID {pid}", (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frames.append(frame)

cap.release()

if frames:
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"Saved to: {output_path}")
else:
    print("No frames to write.")
