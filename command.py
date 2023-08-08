import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

drawing_utils = mp.solutions.drawing_utils
hands_module = mp.solutions.hands

#Gobal variable for mouse drag state
is_holding = False
#Global variable for the threshold angle for a bent finger (adjust as needed)
threshold_angle = 30


# Parameters to send to detect_gesture function
#Configue the time interval between each record button
ACTION_COOLDOWN = 2 
#Holds the last time a button was clicked 
last_action_time = time.time()

capture = cv2.VideoCapture(0)

#Video input's screen dimentions
if capture.isOpened(): 
    width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

#Devices screen dimentions
screen_width, screen_height = pyautogui.size()

#calculates the Euclidean distance between two landmarks in a two-dimensional space
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

#Param: Detected hand landmarks from mediapipe, for for info https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#configurations_options
def detect_gestures(landmarks):
    global is_holding
    global threshold_angle
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
    ring_pip = landmarks.landmark[hands_module.HandLandmark.RING_FINGER_PIP]
    #Pinky
    pinky_tip = landmarks.landmark[hands_module.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks.landmark[hands_module.HandLandmark.PINKY_DIP]
    #Thumb
    thumb_tip = landmarks.landmark[hands_module.HandLandmark.THUMB_TIP]
    thumb_dip = landmarks.landmark[hands_module.HandLandmark.THUMB_IP]

    # Check for gestures
    
    ##############    Pinch gesture detection for dragging capabilities
    # Calculate distances between each finger tip and thumb landmarks
    index_tip_pip_distance = calculate_distance(index_tip, thumb_tip)
    middle_tip_pip_distance = calculate_distance(middle_tip, thumb_tip)
    ring_tip_pip_distance = calculate_distance(ring_tip, thumb_tip)
    pinky_tip_pip_distance = calculate_distance(pinky_tip, thumb_tip)

    # Define distance thresholds for pinch detection (adjust as needed)
    pinch_threshold = 0.05

     # Check if all fingers are pinched
    if ( 
        index_tip_pip_distance < pinch_threshold and
        middle_tip_pip_distance < pinch_threshold and
        ring_tip_pip_distance < pinch_threshold and
        pinky_tip_pip_distance < pinch_threshold
    ):
        #If not holding the button, hold the button
        if(is_holding == False):
            pyautogui.mouseDown(button='left')
            is_holding = True
            return time.time()
    else:
        if(is_holding == True):
            pyautogui.mouseUp(button='left')
            is_holding = False
            return time.time()

    
    ################# Index finger bend down gesture detection for left click
    # Calculate angle between finger and palm using trigonometry
    dx = index_tip.x - index_base.x
    dy = index_tip.y - index_base.y
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Check if finger angle exceeds threshold for left-click
    if angle > threshold_angle:
        pyautogui.click(button='left')
        return time.time()

    # Calculate angle between finger and palm using trigonometry
    dx = middle_tip.x - middle_base.x
    dy = middle_tip.y - middle_base.y
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Check if finger angle exceeds threshold for right-click
    if angle > threshold_angle:
        pyautogui.click(button='right')
        return time.time()
    pyautogui.moveTo(middle_base.x*screen_width,middle_base.y*screen_height)

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