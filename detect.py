import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


video_path = "crash.mp4"


model = YOLO("yolov8m.pt")

# Set the thresholds
stable_threshold = 30
detection_threshold = 0.5
distance_threshold = 20


video = cv2.VideoCapture(video_path)

# Create a deque to store previous centroids
previous_centroids = deque(maxlen=stable_threshold)

while True:
    # Read the next frame from the video
    ret, frame = video.read()
    if not ret:
        break

    # Reset centroids for each frame
    centroids = []

    # Perform object detection using YOLO
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > detection_threshold and class_id == 2:
            # Calculate and store the centroid of the car
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            centroids.append(centroid)

            # Draw the bounding box around the car
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Store the previous centroids
    previous_centroids.append(centroids)

    # Check if the previous centroids are stable
    if len(previous_centroids) == stable_threshold:
        stable = all(np.linalg.norm(np.array(centroid) - np.array(prev_centroid)) < distance_threshold for centroid, prev_centroid in zip(previous_centroids[-1], previous_centroids[0]))
        if stable:
            print("Kaza Algılandı!")
            cv2.putText(frame, "KAZA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        previous_centroids.popleft()


    cv2.imshow("Frame", frame)


    if cv2.waitKey(1) == ord("q"):
        break


video.release()
cv2.destroyAllWindows()
