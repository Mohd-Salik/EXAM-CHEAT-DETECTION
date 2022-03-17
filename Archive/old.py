#DEPENDENCIES
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

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_mesh = mp.solutions.face_mesh
# mp_holistic = mp.solutions.holistic


# def extractKeypoints(results, results_holistic):
#     face = np.array([[[points.x, points.y, points.z] for points in res.landmark] for res in results.multi_face_landmarks]).flatten() if results.multi_face_landmarks else np.zeros(468*3)
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_holistic.pose_landmarks.landmark]).flatten() if results_holistic.pose_landmarks else np.zeros(33*4)
#     return np.concatenate([pose, face])
    

# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(1)
# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as face_mesh:

#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 print("Ignoring empty camera frame.")
#                 # If loading a video, use 'break' instead of 'continue'.
#                 continue
            
            
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(image)
#             results_holistic = holistic.process(image)
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)




#             key_points = extractKeypoints(results, results_holistic)

#             mp_drawing.draw_landmarks(image, results_holistic.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#             if results.multi_face_landmarks:
#                 for face_landmarks in results.multi_face_landmarks:
#                     mp_drawing.draw_landmarks(
#                         image=image,
#                         landmark_list=face_landmarks,
#                         connections=mp_face_mesh.FACEMESH_TESSELATION,
#                         landmark_drawing_spec=None,
#                         connection_drawing_spec=mp_drawing_styles
#                         .get_default_face_mesh_tesselation_style())
#                     mp_drawing.draw_landmarks(
#                         image=image,
#                         landmark_list=face_landmarks,
#                         connections=mp_face_mesh.FACEMESH_CONTOURS,
#                         landmark_drawing_spec=None,
#                         connection_drawing_spec=mp_drawing_styles
#                         .get_default_face_mesh_contours_style())
#                     mp_drawing.draw_landmarks(
#                         image=image,
#                         landmark_list=face_landmarks,
#                         connections=mp_face_mesh.FACEMESH_IRISES,
#                         landmark_drawing_spec=None,
#                         connection_drawing_spec=mp_drawing_styles
#                         .get_default_face_mesh_iris_connections_style())
#             # Flip the image horizontally for a selfie-view display.
#             cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
# cap.release()

#PREPROCESS MOVEMENT DATA SET
DATA_PATH = os.path.join('MOVEMENT_DATA')
actions = np.array(['Gaze Left', 'Gaze Right'])
no_sequences = 15 
sequence_length = 10
start_folder = 20 
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

sequence = []
sentence = []
threshold = 0.8
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]
model.load_weights('facemesh.h5')

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                 
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(frame, results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


#APPLYING HOLISTIC MODELS
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        keypoints = extract_keypoints(frame, results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
            if len(sentence) > 5: 
                sentence = sentence[-5:]
            image = prob_viz(res, actions, image, colors)
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()