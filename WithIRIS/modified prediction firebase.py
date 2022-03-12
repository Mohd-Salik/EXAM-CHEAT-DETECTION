import cv2
import mediapipe as mp
import numpy as np
import os
import time

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import time


actions = np.array(['Left_Head_Tilt',
    'Up_Head_Tilt',
    'Down_Head_Tilt',
    'Right_Head_Tilt',
    'Centered']) 
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1692)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('HeadV1_300_epoch.h5')

colors = [(245,117,16), (117,245,16), (16,117,245), (16,117,245), (16,117,245)]

# def prob_viz(res, actions, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
#         cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
#         print(actions[num], (int(prob*100)))
#     return output_frame

def probabilityDetect(probability_results):
    global time_start, current_action
    for num, prob in enumerate(probability_results):
        # if actions[num] == np.str_()
        action_str = str(actions[num])
        if (int(prob*100)) > 90:
            if time_start == 0:
                time_start = time.time()
                current_action = action_str
            if current_action == action_str:
                pass
            else:
                time_action = time.time() - time_start
                print(current_action, "total time: ", int(time_action))
                time_start = 0
            
        # print(actions[num], (int(prob*100)))


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                 
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def extract_keypoints(results_facemesh, results_holistic):
    face = np.array(getValues(results_facemesh)).flatten() if results_facemesh.multi_face_landmarks else np.zeros(478*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_holistic.pose_landmarks.landmark]).flatten() if results_holistic.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results_holistic.left_hand_landmarks.landmark]).flatten() if results_holistic.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results_holistic.right_hand_landmarks.landmark]).flatten() if results_holistic.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def getValues(results_facemesh):
    final_points = []
    for res in results_facemesh.multi_face_landmarks:
        for points in res.landmark:
            test = np.array([points.x, points.y, points.z])
            final_points.append(test)
    return final_points
    
sequence = []
current_action = ""
time_start = 0
time_action = 0
threshold = 0.8

cap = cv2.VideoCapture(0)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_facemesh = face_mesh.process(image)
            results_holistic = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            key_points = extract_keypoints(results_facemesh, results_holistic)
            
            sequence.append(key_points)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                prediction_results = model.predict(np.expand_dims(sequence, axis=0))[0]
                probabilityDetect(prediction_results)
                # print(actions[np.argmax(res)])
                # image = prob_viz(res, actions, image, colors)

            mp_drawing.draw_landmarks(image, results_holistic.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results_holistic.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results_holistic.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results_facemesh.multi_face_landmarks:
                for face_landmarks in results_facemesh.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
cap.release()