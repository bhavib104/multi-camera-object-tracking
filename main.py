from ultralytics import YOLO
import cv2
import csv
from datetime import datetime

# Load model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("videos/cam1.mp4")

# Dictionary to store object data
tracked_objects = {}

class_names = model.names

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    if results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is not None:
                obj_id = int(box.id)
                cls_id = int(box.cls)
                label = class_names[cls_id]

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # If object seen first time
                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = [label, current_time, current_time]
                else:
                    # Update last seen time
                    tracked_objects[obj_id][2] = current_time

    cv2.imshow("Tracking System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV AFTER processing
with open("database.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Object_ID", "Class", "First_Seen", "Last_Seen"])

    for obj_id, data in tracked_objects.items():
        writer.writerow([obj_id, data[0], data[1], data[2]])

print("✅ Data saved to database.csv")