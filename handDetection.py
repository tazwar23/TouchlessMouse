import os
import uuid
import cv2
import numpy as np
import mediapipe as mp

drawing_utils = mp.solutions.drawing_utils
hands_module = mp.solutions.hands

capture = cv2.VideoCapture(0)

# Parameters for hands object were defined directly for clarity.
with hands_module.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while capture.isOpened():
        read_success, frame = capture.read()

        # Image was converted from BGR to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        detection_results = hands.process(image)
        image.flags.writeable = True

        print(detection_results.multi_hand_landmarks)

        if detection_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_results.multi_hand_landmarks):
                drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=hands_module.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=15),
                    connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=10),
                )

        # Convert the RGB image back to BGR for displaying
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Hand Tracking', image_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
