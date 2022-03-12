from re import S
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.screen import Screen
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.behaviors.magic_behavior import MagicBehavior
from datetime import datetime

import pyrebase
import urllib

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

class MagicButton1(MagicBehavior, MDRaisedButton):
    pass


class visualTracking():
    def __init__(self, **kwargs):
        self.actions = np.array(['Left_Head_Tilt',
            'Up_Head_Tilt',
            'Down_Head_Tilt',
            'Right_Head_Tilt',
            'Centered']) 

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1692)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))
        self.res = [.7, 0.2, 0.1]
        self.model.load_weights('HeadV1_300_epoch.h5')

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.sequence = []
        self.current_action = ""
        self.threshold = 0.8
        self.time_start = 0

        self.final_images = []
        self.final_actions = []
        self.final_time = []

        self.cap = cv2.VideoCapture(1)

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False                 
        results = model.process(image)                 
        image.flags.writeable = True                   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        return image, results

    def probabilityDetect(self, probability_results):
        global time_start, current_action
        for num, prob in enumerate(probability_results):
            # if actions[num] == np.str_()
            action_str = str(self.actions[num])
            if (int(prob*100)) > 90:
                if self.time_start == 0:
                    self.time_start = time.time()
                    self.current_action = action_str
                if self.current_action == action_str:
                    pass
                else:
                    time_action = time.time() - self.time_start
                    print(self.current_action, "total time: ", int(time_action))
                    imagefile = "{0}+'{1}'.png".format(user.studentID(), str(datetime.now()))
                    self.final_images.append(imagefile)
                    self.final_actions.append(self.current_action)
                    self.final_time.append(str(datetime.now()))
                    self.time_start = 0

    def extract_keypoints(self, results_facemesh, results_holistic):
        face = np.array(self.getValues(results_facemesh)).flatten() if results_facemesh.multi_face_landmarks else np.zeros(478*3)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_holistic.pose_landmarks.landmark]).flatten() if results_holistic.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results_holistic.left_hand_landmarks.landmark]).flatten() if results_holistic.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results_holistic.right_hand_landmarks.landmark]).flatten() if results_holistic.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    def getValues(self, results_facemesh):
        final_points = []
        for res in results_facemesh.multi_face_landmarks:
            for points in res.landmark:
                test = np.array([points.x, points.y, points.z])
                final_points.append(test)
        return final_points

    #APPLYING HOLISTIC MODELS
    def runTracking(self):
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

                while self.cap.isOpened():
                    success, image = self.cap.read()
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

                    key_points = self.extract_keypoints(results_facemesh, results_holistic)
                    
                    self.sequence.append(key_points)
                    self.sequence = self.sequence[-30:]
                    if len(self.sequence) == 30:
                        prediction_results = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                        self.probabilityDetect(prediction_results)
                        # print(actions[np.argmax(res)])
                        # image = prob_viz(res, actions, image, colors)

                    self.mp_drawing.draw_landmarks(image, results_holistic.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                    self.mp_drawing.draw_landmarks(image, results_holistic.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                    self.mp_drawing.draw_landmarks(image, results_holistic.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                    if results_facemesh.multi_face_landmarks:
                        for face_landmarks in results_facemesh.multi_face_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles
                                .get_default_face_mesh_tesselation_style())
                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_IRISES,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles
                                .get_default_face_mesh_iris_connections_style())
                    # Flip the image horizontally for a selfie-view display.
                    cv2.imshow('OpenCV Feed', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                self.cap.release()
                cv2.destroyAllWindows()
        result = db.uploadResults(self.final_images, self.final_actions, self.final_time, user.studentID(), user.profMail() ,user.studentRoom())
        print(result)



class databaseInit():
    def __init__(self, **kwargs):
        self.firebaseConfig = {
            'apiKey': "AIzaSyDP7ELddk4JrFfNZJtXcsCgNwrRow_Jw2E",
            'authDomain': "oecd-90c43.firebaseapp.com",
            'projectId': "oecd-90c43",
            'storageBucket': "oecd-90c43.appspot.com",
            'messagingSenderId': "466972383779",
            'appId': "1:466972383779:web:5de25fd2b26c719549ef96",
            'measurementId': "G-X8HMETBY30",
            "databaseURL" : "https://oecd-90c43-default-rtdb.asia-southeast1.firebasedatabase.app/"  
        }
        self.firebase = pyrebase.initialize_app(self.firebaseConfig)
        self.database = self.firebase.database()
        self.auth = self.firebase.auth()

    def signUp(self, email, password):
        try: 
            self.auth.create_user_with_email_and_password(email, password)
            return True
        except:
            return False
    
    def signIn(self, email, password):
        try: 
            self.auth.sign_in_with_email_and_password(email, password)
            return True
        except:
            return False
        
    def createRoom(self, professors_name, room_name):
        name = professors_name.replace(".", "+")
        print("DEBUG: ", name)
        data = {'room name': room_name}
        try: 
            self.database.child("ROOMS").child(name).child(room_name).set(data)
            return True
        except:
            return False

    
    def joinRoom(self, professors_name, room_name):
        name = professors_name.replace(".", "+")
        try:
            rooms = self.database.child("ROOMS").child(name).get()
            for room in rooms.each():
                if room.key()==room_name:
                    return True
        except:
            return False
 
    
    def createStudent(self, studentID, professors_name, room_name):
        name = professors_name.replace(".", "+")
        print("DEBUG: ", name)
        data = {'TIME STARTED: ': str(datetime.now())}
        try: 
            self.database.child("ROOMS").child(name).child(room_name).child(studentID).set(data)
            return True
        except:
            return False

    
    def uploadResults(self, images, actions, time, studentID, professors_name, room_name):
        name = professors_name.replace(".", "+")
        print("DEBUG: ", name)
        data = {}
        for index in range(len(images)):
            data["{}".format(images[index])] = "{}".format(actions[index])
        try: 
            self.database.child("ROOMS").child(name).child(room_name).child(studentID).push(data)
            return True
        except:
            return False

class userInit():
    def __init__(self, **kwargs):
        self.prof_mail = ""
        self.prof_pass = ""
        self.student_name = ""
        self.room_name = ""
    
    def profMail(self):
        return self.prof_mail
    
    def profPass(self):
        return self.prof_pass

    def studentID(self):
        return self.student_name
    
    def studentRoom(self):
        return self.room_name

    def studentLogged(self, identification, professors_email ,room):
        self.student_name = str(identification)
        self.prof_mail = str(professors_email)
        self.room_name = str(room)


    def userLogged(self, email, password):
        self.prof_mail = str(email)
        self.prof_pass = str(password)

    def clearAll(self):
        self.prof_mail = ""
        self.prof_pass = ""
        self.student_name = ""
        self.room_name = ""


class LoginScreen(Screen):
    print("INITIALIZED: LOGIN SCREEN")


class MainStudent(Screen):
    print("INITIALIZED: student SCREEN")

    def joinRoom(self):
        student_name = self.ids.textID_studentname.text
        room_name = self.ids.textID_roomname.text
        prof_mail = self.ids.textID_inputprofmail.text
        
        self.clear()

        if (db.joinRoom(prof_mail, room_name)) == True:
            user.studentLogged(student_name, prof_mail, room_name)
            self.parent.get_screen("kv_LoggedStudent").ids.labelID_loggedstudentlabel.text = "{} have joined the room".format(student_name)
            self.parent.current = "kv_LoggedStudent"

        elif (db.joinRoom(prof_mail, room_name)) == False:
            self.ids.labelID_studentlabel.text = "Invalid/ Cannot Join Room"


    def clear(self):
        self.ids.textID_studentname.text = ""
        self.ids.textID_roomname.text = ""
        self.ids.textID_inputprofmail.text = ""


class MainProfessor(Screen):
    print("INITIALIZED: prof SCREEN")
    
    def logIn(self):
        signin_email = self.ids.textID_profmail.text
        signin_password = self.ids.textID_profpass.text
        self.clear()
        if (db.signIn(signin_email, signin_password)) == True:
            self.ids.labelID_mainprof.text = "SIGN IN SUCCESS"
            user.userLogged(signin_email, signin_password)
            self.parent.current = "kv_LoggedProf"
        elif (db.signIn(signin_email, signin_password)) == False:
            self.ids.labelID_mainprof.text = "SIGN IN FAILED"
    
    def clear(self):
        self.ids.textID_profmail.text = ""
        self.ids.textID_profpass.text = ""


class MainAdmin(Screen):
    print("INITIALIZED: admin SCREEN")


class LoggedProfessor(Screen):
    print("INITIALIZED: LoggedProfessor SCREEN")

    def createRoom(self):
        room_name = self.ids.textID_createroom.text
        self.clear()
        if (room_name) == "":
            self.ids.labelID_loggedprof.text = "Invalid Room Name"
        else:
            if (db.createRoom(user.profMail(), room_name)) == True:
                self.ids.labelID_loggedprof.text = "room '{}' has been created".format(room_name)
            elif (db.createRoom(user.profMail(), room_name)) == False:
                self.ids.labelID_loggedprof.text = "Failed to create room '{}'".format(room_name)
    
    def clear(self):
        self.ids.textID_createroom.text = ""


class LoggedStudent(Screen):
    print("INITIALIZED: LoggedStudent SCREEN")

    def runVisualDetection(self):
        pass

    def createStudentLog(self):
        db.createStudent(user.studentID(), user.profMail() ,user.studentRoom())
        tracking.runTracking()


    

class SignProfessor(Screen):
    print("INITIALIZED: Sign-Up SCREEN")


    def signUpProcess(self):
        signup_email = self.ids.textID_signprofmail.text
        signup_password = self.ids.textID_signprofpass.text
        self.clear()
        if (db.signUp(signup_email, signup_password)) == True:
            self.parent.get_screen("kv_Signed").ids.labelID_signed.text = "ACCOUNT HAS BEEN CREATED, YOU MAY NOW LOG IN"
        elif (db.signUp(signup_email, signup_password)) == False:
            self.parent.get_screen("kv_Signed").ids.labelID_signed.text = "SIGN UP FAILED"
            

    def clear(self):
        self.ids.textID_signprofmail.text = ""
        self.ids.textID_signprofpass.text = ""


class Signed(Screen):
    print("INITIALIZED: Sign-Up Sucess SCREEN")


class MyRooms(Screen):
    print("INITIALIZED: MyRooms SCREEN")

    def displayRooms(self):
        pass


# Main build class
class OECP(MDApp):
    def build(self):
        global sm
        self.load_kv('main.kv')
        sm = ScreenManager()
        sm.add_widget(MainStudent(name = 'kv_MainStudent'))
        sm.add_widget(MyRooms(name = 'kv_MyRooms'))
        sm.add_widget(LoggedProfessor(name = 'kv_LoggedProf'))
        sm.add_widget(MainStudent(name = 'kv_MainStudent'))
        sm.add_widget(MainProfessor(name = 'kv_MainProf'))
        sm.add_widget(SignProfessor(name = 'kv_SignProf'))
        sm.add_widget(Signed(name = 'kv_Signed'))
        sm.add_widget(LoginScreen(name = 'kv_login'))
        sm.add_widget(LoggedStudent(name = 'kv_LoggedStudent'))
        
        sm.add_widget(MainAdmin(name = 'kv_MainAdmin'))
        print("INITIALIZED: SCREEN MANAGER AND SCREENS")
        return sm


if __name__ == "__main__":
    # # Kivy Initialization

    print("INITIALIZED: MAIN")
    db = databaseInit()
    user = userInit()
    tracking = visualTracking()
    Window.size = (600, 300)
    OECP().run()

