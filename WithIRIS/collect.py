#DEPENDENCIES
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#APPLYING HOLISTIC MODELS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)

DATA_PATH = os.path.join('FINAL_TRAINING_HEAD') # Path for exported data, numpy arrays
actions = np.array(['Left_Head_Tilt',
    'Up_Head_Tilt',
    'Down_Head_Tilt',
    'Right_Head_Tilt',
    'Centered']) 
no_sequences = 30 # Thirty videos worth of data
sequence_length = 30 # Videos are going to be 30 frames in length

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

def extract_keypoints(results_facemesh, results_holistic):
    # face = np.array([[[point.x, point.y, point.z] for point in res.landmark] for res in results_facemesh.multi_face_landmarks]) if results_facemesh.multi_face_landmarks else np.zeros(1434*3)
    face = np.array(getValues(results_facemesh)).flatten() if results_facemesh.multi_face_landmarks else np.zeros(478*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_holistic.pose_landmarks.landmark]).flatten() if results_holistic.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results_holistic.left_hand_landmarks.landmark]).flatten() if results_holistic.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results_holistic.right_hand_landmarks.landmark]).flatten() if results_holistic.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
    # return pose
    # if results_facemesh.multi_face_landmarks:
    #     # print("FACE: ", type(results_facemesh.multi_face_landmarks))
    #     for res in results_facemesh.multi_face_landmarks:
    #         print(res.landmark)
            # for points in res.landmark:
            #     print(points)
    # if results_holistic.pose_landmarks:
    #     print("OTHER: ", type(results_holistic.pose_landmarks))
    #     for res in results_holistic.pose_landmarks.landmark:
    #             # for points in res.landmark:
    #         print(res.x)
    # print("FACE TYPE: ", type(face))
    # print("FACE SHAPE: ", face.shape)
    # print("FACE DTYPE: ", type(face.dtype))

    # print("POSE TYPE: ", type(pose))
    # print("POSE SHAPE: ", pose.shape)
    # print("POSE DTYPE: ", type(pose.dtype))

    # print("LH TYPE: ", type(lh))
    # print("LH SHAPE: ", lh.shape)
    # print("LH DTYPE: ", type(lh.dtype))
    # print("-*-"*100)
  
   


def getValues(results_facemesh):
    final_points = []
    for res in results_facemesh.multi_face_landmarks:
        for points in res.landmark:
            test = np.array([points.x, points.y, points.z])
            final_points.append(test)
    return final_points


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    ret, image = cap.read()
                    # image = cv2.imread("11.jpg")
                    # Make detections
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results_facemesh = face_mesh.process(image)
                    results_holistic = holistic.process(image)
                    # Draw the face mesh annotations on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    mp_drawing.draw_landmarks(image, results_holistic.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
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
                    
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'Collecting..', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, 'frame.. {}'.format(frame_num), (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # NEW Export keypoints
                    key_points = extract_keypoints(results_facemesh, results_holistic)
                    # print("KEYPOINTS TYPE: ", type(key_points))
                    # print("KEYPOINTS SHAPE: ", key_points.shape)
                    # print("KEYPOINTS DTYPE: ", type(key_points.dtype))
                    # print("-*-"*100)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, key_points)
                    print("KEYPOINTS TYPE: ", type(key_points))
                    print("KEYPOINTS SHAPE: ", key_points.shape)
                    print("KEYPOINTS DTYPE: ", type(key_points.dtype))
                    print("SAVED","-*-"*20)
                    # # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break        
            
        

cap.release()
cv2.destroyAllWindows()
