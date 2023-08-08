import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

drawing_utils = mp.solutions.drawing_utils
hands_module = mp.solutions.hands

# Parameters
ACTION_COOLDOWN = 2  # seconds
last_action_time = time.time()

capture = cv2.VideoCapture(0)

if capture.isOpened(): 
    width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

screen_width, screen_height = pyautogui.size()

def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)


def detect_gestures(landmarks):
    # Compute required landmarks once
    #Index Finger
    index_tip = landmarks.landmark[hands_module.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks.landmark[hands_module.HandLandmark.INDEX_FINGER_PIP]
    index_dip = landmarks.landmark[hands_module.HandLandmark.INDEX_FINGER_DIP]
    index_base = landmarks.landmark[hands_module.HandLandmark.INDEX_FINGER_MCP]
    #Middle Finger
    middle_tip = landmarks.landmark[hands_module.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks.landmark[hands_module.HandLandmark.MIDDLE_FINGER_PIP]
    middle_dip = landmarks.landmark[hands_module.HandLandmark.MIDDLE_FINGER_DIP]
    middle_base = landmarks.landmark[hands_module.HandLandmark.MIDDLE_FINGER_MCP]
    #Ringer Finger
    ring_tip = landmarks.landmark[hands_module.HandLandmark.RING_FINGER_TIP]
    #Pinky
    pinky_tip = landmarks.landmark[hands_module.HandLandmark.PINKY_TIP]
    #Thumb
    thumb_tip = landmarks.landmark[hands_module.HandLandmark.THUMB_TIP]
    thumb_dip = landmarks.landmark[hands_module.HandLandmark.THUMB_DIP]

    
    
    
    

    

    # Check for gestures

    # Mouse movement
     # Calculate distances between tip and fingertip landmarks
    thumb_tip_dip_distance = calculate_distance(thumb_tip, thumb_dip)
    index_tip_pip_distance = calculate_distance(index_tip, index_pip)
    middle_tip_pip_distance = calculate_distance(middle_tip, landmarks.landmark[hands_module.HandLandmark.MIDDLE_FINGER_PIP])
    ring_tip_pip_distance = calculate_distance(ring_tip, landmarks.landmark[hands_module.HandLandmark.RING_FINGER_PIP])
    pinky_tip_pip_distance = calculate_distance(pinky_tip, landmarks.landmark[hands_module.HandLandmark.PINKY_DIP])

    # Define distance thresholds for pinch detection (adjust as needed)
    pinch_threshold = 0.02

     # Check if all fingers are pinched
    if (
        thumb_tip_dip_distance < pinch_threshold and
        index_tip_pip_distance < pinch_threshold and
        middle_tip_pip_distance < pinch_threshold and
        ring_tip_pip_distance < pinch_threshold and
        pinky_tip_pip_distance < pinch_threshold
    ):
        pyautogui.mouseDown(button='left')
    else:
        pyautogui.mouseUp(button='left')

    # Left click
    # Calculate angle between finger and palm using trigonometry
    dx = index_tip.x - index_base.x
    dy = index_tip.y - index_base.y
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Define the threshold angle for a bent finger (adjust as needed)
    threshold_angle = 30
    if angle > threshold_angle:
        pyautogui.click(button='left')
        return time.time()

    dx = middle_tip.x - middle_base.x
    dy = middle_tip.y - middle_base.y
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Check if finger angle exceeds threshold for right-click
    if angle > threshold_angle:
        pyautogui.click(button='right')
        return time.time()
    pyautogui.moveTo(middle_tip.x*screen_width,middle_tip.y*screen_height)

    return last_action_time


with hands_module.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while capture.isOpened():
        read_success, frame = capture.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image,1)
        image.flags.writeable = False
        detection_results = hands.process(image)
        image.flags.writeable = True

        if detection_results.multi_hand_landmarks:
            for landmarks in detection_results.multi_hand_landmarks:
                drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=landmarks,
                    connections=hands_module.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=15),
                    connection_drawing_spec=drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=10)
                )

                if time.time() - last_action_time > ACTION_COOLDOWN:
                    last_action_time = detect_gestures(landmarks)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Tracking', image_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()