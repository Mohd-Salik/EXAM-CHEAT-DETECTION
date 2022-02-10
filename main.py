print("program started")

#INSTALLING DEPENDENCIES
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


#GETTING REAL TIME CAMERA 
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     cv2.imshow('Holistic Model Detections', frame)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



#APPLYING HOLISTIC MODELS
cap = cv2.VideoCapture(0)

#INIT
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()


        #RECOLOR FEED
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #MAKE DETECTIONS
        results = holistic.process(image)

        #face_Landmarks, pose_Landmarks, left_hand_Landmarks, right_hand_Landmarks
        # print(results.face_landmarks)
        
        #DRAWING ON FEED FACEMESH_TESSELATION, FACEMESH_CONTOURS
        # mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

        #DRAWING RIGHT HAND
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0,0,240), thickness=2, circle_radius=2)
        )

        #DRAWING LEFT HAND
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
        )

        #DRAWING POSE DETECTION
        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('THESIS FACE DETECTION TESTING', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()



#apply styling