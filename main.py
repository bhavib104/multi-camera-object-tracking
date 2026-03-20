from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Load both videos
cap1 = cv2.VideoCapture("videos/cam1.mp4")
cap2 = cv2.VideoCapture("videos/cam2.mp4")

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Cannot open videos")
    exit()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("End of one video")
        break

    # Run detection + tracking
    results1 = model.track(frame1, persist=True)
    results2 = model.track(frame2, persist=True)

    # Annotate frames
    annotated1 = results1[0].plot()
    annotated2 = results2[0].plot()

    # Resize for side-by-side view
    annotated1 = cv2.resize(annotated1, (640, 360))
    annotated2 = cv2.resize(annotated2, (640, 360))

    # Combine frames
    combined = cv2.hconcat([annotated1, annotated2])

    # Show output
    cv2.imshow("Multi-Camera Tracking", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()