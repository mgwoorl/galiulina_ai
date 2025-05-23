import math
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def calculate_angle(point_a, point_b, point_c):
    angle_start = np.arctan2(point_c[1] - point_b[1], point_c[0] - point_b[0])
    angle_end = np.arctan2(point_a[1] - point_b[1], point_a[0] - point_b[0])
    angle_deg = np.rad2deg(angle_start - angle_end)
    angle_deg = angle_deg + 360 if angle_deg < 0 else angle_deg
    return 360 - angle_deg if angle_deg > 180 else angle_deg

def process_pose(frame, keypoints):
    nose_visible = keypoints[0][0] > 0 and keypoints[0][1] > 0
    left_ear_visible = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_ear_visible = keypoints[4][0] > 0 and keypoints[4][1] > 0

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    right_elbow = keypoints[7]
    left_elbow = keypoints[8]
    right_wrist = keypoints[9]
    left_wrist = keypoints[10]

    try:
        if left_ear_visible and not right_ear_visible:
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            x, y = int(left_elbow[0]), int(left_elbow[1])
        else:
            elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            x, y = int(right_elbow[0]), int(right_elbow[1])

        cv2.putText(frame, f"{int(elbow_angle)}", (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 25, 255), 2)

        return elbow_angle

    except ZeroDivisionError:
        return None

model_path = "yolo11n-pose.pt"
pose_model = YOLO(model_path)

cap = cv2.VideoCapture(0)
video_writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"avc1"), 10, (640, 480))

last_frame_time = time.time()
rep_started = False
repetition_count = 0
angle_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    video_writer.write(frame)

    current_time = time.time()
    fps = 1 / (current_time - last_frame_time)
    last_frame_time = current_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 255, 25), 1)
    cv2.imshow('YOLO', frame)

    results = pose_model(frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    if not results:
        continue

    result = results[0]
    keypoints_xy = result.keypoints.xy.tolist()
    if not keypoints_xy:
        continue

    keypoints = keypoints_xy[0]
    if not keypoints:
        continue

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated_frame = annotator.result()

    angle_value = process_pose(annotated_frame, keypoints)

    if angle_value is not None:
        if len(angle_history) >= 5:
            angle_history.append(angle_value)
            angle_history.pop(0)

            average_angle = sum(angle_history) / len(angle_history)

            if rep_started and average_angle >= 100:
                repetition_count += 1
                rep_started = False
            elif average_angle < 100:
                rep_started = True
        else:
            angle_history.append(angle_value)

    cv2.putText(annotated_frame, f"Count: {repetition_count}", (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 255, 25), 1)
    cv2.imshow("Pose", annotated_frame)

video_writer.release()
cap.release()
cv2.destroyAllWindows()

