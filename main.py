#DEPENDENCIES
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#APPLYING HOLISTIC MODELS
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(1)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        results = holistic.process(frame)
        draw_landmarks(frame, results)
        cv2.imshow('Test', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



#apply styling

#DRAWING ON FEED FACEMESH_TESSELATION, FACEMESH_CONTOURS
# mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

# #DRAWING RIGHT HAND
# mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
# mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
# mp_drawing.DrawingSpec(color=(0,0,240), thickness=2, circle_radius=2)
# )

# #DRAWING LEFT HAND
# mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
# mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
# mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
# )