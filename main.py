from re import S
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.screen import Screen
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.behaviors.magic_behavior import MagicBehavior

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
        self.DATA_PATH = os.path.join('MOVEMENT_DATA')
        self.actions = np.array(['Gaze Normal', 'Gaze Left', 'Gaze Right'])
        self.no_sequences = 30 
        self.sequence_length = 30
        self.start_folder = 30 
        self.label_map = {label:num for num, label in enumerate(self.actions)}
        self.sequences, self.labels = [], []
        for action in self.actions:
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                self.sequences.append(window)
                self.labels.append(self.label_map[action])

        sequence = []
        sentence = []
        threshold = 0.8
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))
        self.res = [.7, 0.2, 0.1]
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.load_weights('movement.h5')

        self.colors = [(245,117,16), (117,245,16), (16,117,245)]

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.cap = cv2.VideoCapture(1)

        self.sequence = []
        self.sentence = []
        self.threshold = 0.8

        self.cap = cv2.VideoCapture(0)

    def prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False                 
        results = model.process(image)                 
        image.flags.writeable = True                   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        return image, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    #APPLYING HOLISTIC MODELS
    def runTracking(self):
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as self.holistic:
            while self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                self.image, self.results = self.mediapipe_detection(self.frame, self.holistic)
                print(self.results)
                self.draw_landmarks(self.image, self.results)
                self.keypoints = self.extract_keypoints(self.results)
                self.sequence.append(self.keypoints)
                self.sequence = self.sequence[-30:]
                if len(self.sequence) == 30:
                    self.res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    print(self.actions[np.argmax(self.res)])
                    if self.res[np.argmax(self.res)] > self.threshold: 
                        if len(self.sentence) > 0: 
                            if self.actions[np.argmax(self.res)] != self.sentence[-1]:
                                self.sentence.append(self.actions[np.argmax(self.res)])
                        else:
                            self.sentence.append(self.actions[np.argmax(self.res)])
                    if len(self.sentence) > 5: 
                        self.sentence = self.sentence[-5:]
                    self.image = self.prob_viz(self.res, self.actions, self.image, self.colors)
                cv2.imshow('OpenCV Feed', self.image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()



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
            self.database.child("ROOMS").child(name).push(data)
            return True
        except:
            return False

    
    def joinRoom(self, professors_name, room_name):
        name = professors_name.replace(".", "+")
        try:
            rooms = self.database.child("ROOMS").child(name).get()
            for room in rooms.each():
                if room.val()['room name']==room_name:
                    return True
        except:
            return False
 

class userInit():
    def __init__(self, **kwargs):
        self.user_mail = ""
        self.user_pass = ""
    
    def userMail(self):
        return self.user_mail
    
    def userPass(self):
        return self.user_pass

    def userLogged(self, email, password):
        self.user_mail = str(email)
        self.user_pass = str(password)


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
            if (db.createRoom(user.userMail(), room_name)) == True:
                self.ids.labelID_loggedprof.text = "room '{}' has been created".format(room_name)
            elif (db.createRoom(user.userMail(), room_name)) == False:
                self.ids.labelID_loggedprof.text = "Failed to create room '{}'".format(room_name)
    
    def clear(self):
        self.ids.textID_createroom.text = ""


class LoggedStudent(Screen):
    print("INITIALIZED: LoggedStudent SCREEN")

    def runVisualDetection(self):
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


# Main build class
class OECP(MDApp):
    def build(self):
        global sm
        self.load_kv('main.kv')
        sm = ScreenManager()
        sm.add_widget(MainStudent(name = 'kv_MainStudent'))
        sm.add_widget(MainProfessor(name = 'kv_MainProf'))
        sm.add_widget(SignProfessor(name = 'kv_SignProf'))
        sm.add_widget(Signed(name = 'kv_Signed'))
        sm.add_widget(LoginScreen(name = 'kv_login'))
        sm.add_widget(LoggedStudent(name = 'kv_LoggedStudent'))
        sm.add_widget(LoggedProfessor(name = 'kv_LoggedProf'))
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

