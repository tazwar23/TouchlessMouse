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

print(width)
print(height)
print(screen_width)
print(screen_height)

capture.release()
cv2.destroyAllWindows()