from cv2 import threshold
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.screen import Screen
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.behaviors.magic_behavior import MagicBehavior
from kivymd.uix.list import OneLineAvatarListItem, ThreeLineAvatarListItem, TwoLineAvatarListItem, ImageLeftWidget


import pyrebase
import urllib

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime

from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import TensorBoard

class MagicButton1(MagicBehavior, MDRaisedButton):
    pass


class visualTracking():
    def __init__(self, **kwargs):
        self.actions = np.array(['Left_Head_Tilt',
            'Up_Head_Tilt',
            'Down_Head_Tilt',
            'Right_Head_Tilt',
            'Centered']) 

        # Create and load LSTM model
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1692)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))
        self.res = [.7, 0.2, 0.1]
        self.model.load_weights('HeadV1_300_epoch.h5')

        # Initialize Mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh

        self.cap = cv2.VideoCapture(1)
        self.sequence = []
        self.current_action = ""
        self.threshold = 0.8
        self.time_start = 0

        self.final_images = [] # ["date]studentid", "date]studentid"]
        self.final_actions = [] # ["lookleft", "lookright"]
        self.final_time = [] # ["8", "7", "4"]
        self.imagefile = ""
        self.save = False



    def probabilityDetect(self, probability_results, image, end):
        global time_start, current_action
        for num, prob in enumerate(probability_results):
            action_str = str(self.actions[num])
            
            # If the action has 90% detection probability
            if (int(prob*100)) > 80:
                if (action_str == "Up_Head_Tilt"):
                    pass
                
                else:
                    if end == True:
                        time_action = int(time.time() - self.time_start)
                        self.final_time.append(str(time_action))
                        self.time_start = 0
                        self.save = False
                        print(self.current_action, "total time: ", int(time_action))

                    else:
                    # Start timer and extract current image frame as jpg
                        if self.time_start == 0:
                            self.time_start = time.time()
                            self.current_action = action_str
                            
                        if self.current_action == action_str:
                            if self.save == False:
                                self.save = True
                                self.final_actions.append(self.current_action)
                                date = str(datetime.now()).replace(".", "-").replace(":", "-").replace(" ", "-")
                                self.imagefile = "{}]{}.jpg".format(date, user.studentID())
                                try:
                                    cv2.imwrite("Imagebin\\{}".format(self.imagefile), image)
                                    self.final_images.append(self.imagefile)
                                    print("SUCESS: Image was saved: ", self.current_action)
                                except:
                                    print("ERROR: Could not save frame from cheating movement")
                                
                        # If the action detected is not the previous action, save all results and reset timer
                        else:
                            time_action = int(time.time() - self.time_start)
                            self.final_time.append(str(time_action))
                            self.time_start = 0
                            self.save = False

                            print(self.current_action, "total time: ", int(time_action))
            # else:
            #     time_action = int(time.time() - self.time_start)
            #     self.final_time.append(str(time_action))
            #     self.time_start = 0
            #     self.save = False

            #     print(self.current_action, "total time: ", int(time_action))



    # Return all keypoints from the frame
    def extract_keypoints(self, results_facemesh, results_holistic):
        face = np.array(self.getValues(results_facemesh)).flatten() if results_facemesh.multi_face_landmarks else np.zeros(478*3)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_holistic.pose_landmarks.landmark]).flatten() if results_holistic.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results_holistic.left_hand_landmarks.landmark]).flatten() if results_holistic.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results_holistic.right_hand_landmarks.landmark]).flatten() if results_holistic.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    

    # Return keypoints from face_mesh solutions (Budfix)
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
                        self.probabilityDetect(prediction_results, image, True)
                        break
                    
                    # Converting image frame to readable and writable
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results_facemesh = face_mesh.process(image)
                    results_holistic = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Analyzing frame and predicting the aciton every 30 frames
                    key_points = self.extract_keypoints(results_facemesh, results_holistic)
                    self.sequence.append(key_points)
                    self.sequence = self.sequence[-30:]
                    if len(self.sequence) == 30:
                        prediction_results = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                        self.probabilityDetect(prediction_results, image, False)

                    # Draw preview of the keypoints
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
                    cv2.imshow('Realtime Detection', image)
                    
                    # Quit the detection window
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.probabilityDetect(prediction_results, image, True)
                        break
                
                self.cap.release()
                cv2.destroyAllWindows()
        
        # Upload Results to database
        result = db.uploadResults(
            self.final_images, 
            self.final_actions, 
            self.final_time, 
            user.studentID(), 
            user.profMail(), 
            user.studentRoom())
        print("RESULTS: ", result)


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
        self.storage = self.firebase.storage()

    # Signin/Signup for professors
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

    # Create examination room on the professors name
    def createRoom(self, professors_name, room_name):
        name = professors_name.replace(".", "+")
        print("DEBUG: ", name)
        data = {'room name': room_name}
        try: 
            self.database.child("ROOMS").child(name).child(room_name).set(data)
            return True
        except:
            return False

    # Verify if such exam room exist on the professors name
    def joinRoom(self, professors_name, room_name):
        name = professors_name.replace(".", "+")
        try:
            rooms = self.database.child("ROOMS").child(name).get()
            for room in rooms.each():
                if room.key()==room_name:
                    return True
        except:
            return False

    def getRooms(self, profname):
        prof_name = profname.replace(".", "+")
        data = self.database.child("ROOMS").child(prof_name).get()
        total = []
        for person in data.each():
            total.append(person.val()['room name'])
        return total
    
    def getStudents(self, profname, room_name):
        prof_name = profname.replace(".", "+")
        data = self.database.child("ROOMS").child(prof_name).child(room_name).get()
        total = []
        for person in data.each():
            if (str(person.key()) == "room name"):
                pass
            total.append(str(person.key()))
        return total

    def getStudentData(self, profname, room_name, student_name):
        prof_name = profname.replace(".", "+")
        print("DATABSE: ", prof_name, room_name, student_name)
        data = self.database.child("ROOMS").child(prof_name).child(room_name).child(student_name).get()
        print("DATA: ", data)
        actions = []
        images = []
        for entry in data.each():
            try:
                actions.append(entry.val()["Action"])
                images.append(entry.val()["File"])
            except:
                print("DEBUG DATA: NOT AN ENTRY")

        return actions, images
    
    def getImage(self, prof_name, room_name, student_name, image_name):
        prof_name_db = prof_name.replace(".", " ")
        path = "ROOMS/{}/{}/{}/{}".format(prof_name_db, room_name, student_name, image_name)
        print(path)
        try:
            self.storage.child(path).download("", "Imagebin/{}".format(image_name))
            print("DOWNLAODED IMAGE")
            return True
        except:
            return False

    # Create student entry on the examination room
    def createStudent(self, studentID, professors_name, room_name):
        name = professors_name.replace(".", "+") # sample@gmail+com (Bugfix)
        data = {'TIME STARTED: ': str(datetime.now())}
        try: 
            self.database.child("ROOMS").child(name).child(room_name).child(studentID).set(data)
            return True
        except:
            return False

    # Upload detection results to student entry
    def uploadResults(self, images, actions, time, studentID, professors_name, room_name):
        name = professors_name.replace(".", "+")

        for index in range(len(images)):
            try:
                action = "{}]{}".format(actions[index], time[index])
                data = {"Action": action, "File" : images[index]}
                print("APPENDING: ", data)
            except:
                print("ERROR: Index Failed")

            try: 
                self.storage.child("ROOMS").child(name).child(room_name).child(studentID).child(images[index]).put("Imagebin/"+images[index])
                self.database.child("ROOMS").child(name).child(room_name).child(studentID).push(data)
            except:
                print("ERROR: Cannot push data to student entry")

        Image_dir = "Imagebin"
        for f in os.listdir(Image_dir):
            os.remove(os.path.join(Image_dir, f))


class userInit():
    def __init__(self, **kwargs):
        self.prof_mail = "test_professor@gmail.com"
        self.prof_pass = "test_professor"
        self.student_name = ""
        self.room_name = ""
        self.actions = []
        self.threshold = "MODERATE"
    
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

    def clearAll(self):
        self.prof_mail = ""
        self.prof_pass = ""
        self.student_name = ""
        self.room_name = ""

    def setRoomName(self, room):
        self.room_name = str(room)

    def getRoomName(self):
        return self.room_name

    def clearRoomName(self):
        self.room_name = ""

    def setStudentName(self, student):
         self.student_name = str(student)

    def getStudentName(self):
        return  self.student_name

    def clearStudentName(self):
        self.student_name = ""

    def setProfessorName(self, prof_name):
        self.prof_mail = str(prof_name)

    def getProfessorName(self):
        return self.prof_mail

    def setProfessorPassword(self, prof_pass):
        self.prof_pass = str(prof_pass)

    def setResults(self, actions):
        self.actions = actions

    def getResults(self):
        return self.actions

    def setThreshold(self, threshold):
        self.threshold = str(threshold)

    def getThreshold(self):
        return self.threshold


class CalculatePrediction():
    
    def getPrediction(threshold, percentage):
        red = "massive movements \ndetected: cheating"
        yellow = "minor movements \ndetected: probability of cheating"
        green = "no major movements \ndetected: normal feedback"

        if threshold == "STRICT":
            if 98 <= percentage <= 100:
                return "green", green
            elif 95 <= percentage <= 97:
                return "yellow", yellow
            elif percentage < 95:
                return "red", red
        elif threshold == "MODERATE":
            if 90 <= percentage <= 100:
                return "green", green
            elif 80 <= percentage <= 89:
                return "yellow", yellow
            elif percentage < 80:
                return "red", red
        elif threshold == "LENIENT":
            if 80 <= percentage <= 100:
                return "green", green
            elif 70 <= percentage <= 79:
                return "yellow", yellow
            elif percentage < 70:
                return "red", red

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
            
            try:
                Image_dir = "Imagebin"
                for f in os.listdir(Image_dir):
                    os.remove(os.path.join(Image_dir, f))
            except:
                print("DEBUG: CANNOT DELETE IMAGES")

            self.parent.current = "kv_LoggedStudent"

        elif (db.joinRoom(prof_mail, room_name)) == False:
            self.ids.labelID_studentlabel.text = "Invalid/ Cannot Join Room"


    def clear(self):
        self.ids.textID_studentname.text = ""
        self.ids.textID_roomname.text = ""
        self.ids.textID_inputprofmail.text = ""


class MainProfessor(Screen):

    def logIn(self):
        signin_email = self.ids.textID_profmail.text
        signin_password = self.ids.textID_profpass.text

        if (db.signIn(signin_email, signin_password)) == True:
            print("DEBUG: {}, {} Successfully Logged".format(signin_email, signin_password))
            user.setProfessorName(signin_email)
            user.setProfessorPassword(signin_password)

            self.ids.labelID_mainprof.text = "SIGN IN SUCCESS"
            self.parent.get_screen("kv_LoggedProf").ids.labelID_loggedprof.text = "Welcome {}!".format(signin_email)
            self.parent.current = "kv_LoggedProf"

        elif (db.signIn(signin_email, signin_password)) == False:
            print("DEBUG: {}, {} Log In Failed, Invalid Input".format(signin_email, signin_password))
            self.ids.labelID_mainprof.text = "SIGN IN FAILED"
        
        self.clear()

    def clear(self):
        self.ids.textID_profmail.text = ""
        self.ids.textID_profpass.text = ""

    def resetText(self):
        self.ids.labelID_mainprof.text = 'LOGIN PAGE'

    


class MainAdmin(Screen):
    print("DEBUG: Main Admin Screen")


class LoggedProfessor(Screen):

    def createRoom(self):
        room_name = self.ids.textID_createroom.text
        if (room_name) == "":
            self.ids.labelID_loggedprof.text = "Invalid: Room Name Empty"

        else:
            if (db.createRoom(user.profMail(), room_name)) == True:
                self.ids.labelID_loggedprof.text = "Room '{}'\n has been created\n".format(room_name)
                print("DEBUG: Created Room {}".format(room_name))

            elif (db.createRoom(user.profMail(), room_name)) == False:
                self.ids.labelID_loggedprof.text = "Failed to create room '{}'".format(room_name)
                print("DEBUG: Failed to Create Room {}".format(room_name))

        self.clear()

    def clear(self):
        self.ids.textID_createroom.text = ""
    
    def loadRooms(self):
        list_rooms = db.getRooms(user.getProfessorName())

        for room in list_rooms:
            room_name = str(room)
            list_item = TwoLineAvatarListItem(text = room_name, on_release = lambda room_name:self.loadStudents(room_name))
            self.parent.get_screen("kv_MyRooms").ids.listID_MainList.add_widget(list_item)
        print("DEBUG: Rooms for {} Loaded, {}".format(user.getProfessorName(), list_rooms))
    
    def loadStudents(self, room):
        global sm
        room_name = room.text
        user.setRoomName(room_name)
        sm.get_screen("kv_MyRooms").ids.buttonID_CreateSummary.disabled = False
        sm.get_screen("kv_MyRooms").ids.listID_MainList.clear_widgets()
        list_students = db.getStudents(user.getProfessorName(), str(room_name))

        for students in list_students:
            students_name = str(students)
            profile = TwoLineAvatarListItem(text = students_name, on_release = lambda students_name:self.loadData(students_name))
            sm.get_screen("kv_MyRooms").ids.listID_MainList.add_widget(profile)
        print("DEBUG: Loaded All Students on Room")
        
    def loadData(self, student_name):
        global sm
        user.setStudentName(student_name.text)
        list_actions, list_images = db.getStudentData(user.getProfessorName(), user.getRoomName(), student_name.text)
        
        # Calculating Variation Percentage
        total_left = []
        total_right = []
        total_center = []
        total_lower = []
        
        for actions in list_actions:
            actions_split = actions.split("]")
            if actions_split[0] == "Right_Head_Tilt":
                total_right.append(int(actions_split[1]))
            elif actions_split[0] == "Left_Head_Tilt":
                total_left.append(int(actions_split[1]))
            elif actions_split[0] == "Down_Head_Tilt":
                total_lower.append(int(actions_split[1]))
            elif actions_split[0] == "Centered":
                total_center.append(int(actions_split[1]))
            else:
                print("ACTION NOT FOUND: ", actions_split)
        total_detection = len(total_center) + len(total_right) + len(total_lower) + len(total_left)
        summation = sum(total_center) + sum(total_right) + sum(total_lower) + sum(total_left)
        all_results = {
        "Centered" : round(((sum(total_center)/summation)*100), 2),
        "Left_Head_Tilt" : round(((sum(total_left)/summation)*100), 2),
        "Right_Head_Tilt" : round(((sum(total_right)/summation)*100), 2),
        "Down_Head_Tilt" : round(((sum(total_lower)/summation)*100), 2),
        }
        highest = max(all_results, key=all_results.get)

        color, prediction_results = CalculatePrediction.getPrediction(user.getThreshold(), all_results[highest])
        summary = "PREDICTION: {}\nTOTAL DETECTION: {}\nCENTERED: {}%\nLEFT TILT: {}%\n RIGHT TILT: {}%\n LOWER TILT: {}%".format(
            prediction_results,
            total_detection,
            all_results["Centered"],
            all_results["Left_Head_Tilt"],
            all_results["Right_Head_Tilt"],
            all_results["Down_Head_Tilt"]
        )
        sm.get_screen("kv_MyData").ids.labelID_DataLabel.text = summary
        user.setResults(list_actions)
        print("DEBUG: Calculated Percentage Variations")
        
        for image in list_images:
            image_file = str(image)
            action_name = list_actions[list_images.index(image_file)]
            action_split = action_name.split("]")
            if color == "green":
                profile = TwoLineAvatarListItem(text = image_file, secondary_text = "{} for {} seconds".format(action_split[0], action_split[1]), on_release = lambda image_file:self.loadImage(image_file))
            else:
                if action_split[0] == highest:
                    profile = TwoLineAvatarListItem(text = image_file, secondary_text = "{} for {} seconds".format(action_split[0], action_split[1]), on_release = lambda image_file:self.loadImage(image_file))
                else:
                    if color == "red":
                        profile = TwoLineAvatarListItem(text = image_file, secondary_text = "{} for {} seconds".format(action_split[0], action_split[1]), on_release = lambda image_file:self.loadImage(image_file))
                        profile.add_widget(ImageLeftWidget(source = "UI/red.png"))
                    elif color == "yellow":
                        profile = TwoLineAvatarListItem(text = image_file, secondary_text = "{} for {} seconds".format(action_split[0], action_split[1]), on_release = lambda image_file:self.loadImage(image_file))
                        profile.add_widget(ImageLeftWidget(source = "UI/yellow.png"))
            sm.get_screen("kv_MyData").ids.listID_DataList.add_widget(profile)
        sm.current = "kv_MyData"

    def loadImage(self, image_file):
        global sm
        status = db.getImage(user.getProfessorName(), user.getRoomName(), user.getStudentName(), image_file.text)
        if status == True:
            sm.get_screen("kv_MyData").ids.imageID_DataImage.source = "Imagebin/{}".format(image_file.text)
        elif status == False: 
            sm.get_screen("kv_MyData").ids.imageID_DataImage.source = "UI/default.jpg"


class LoggedStudent(Screen):

    def runVisualDetection(self):
        pass

    def createStudentLog(self):
        db.createStudent(user.studentID(), user.getProfessorName() , user.studentRoom())
        tracking.runTracking()


class SignProfessor(Screen):

    def signUpProcess(self):
        signup_email = self.ids.textID_signprofmail.text
        signup_password = self.ids.textID_signprofpass.text
        self.clear()

        if (db.signUp(signup_email, signup_password)) == True:
            self.parent.get_screen("kv_Signed").ids.labelID_signed.text = "ACCOUNT HAS BEEN CREATED\nYOU MAY NOW LOG IN"
        elif (db.signUp(signup_email, signup_password)) == False:
            self.parent.get_screen("kv_Signed").ids.labelID_signed.text = "SIGN UP FAILED"
            
    def clear(self):
        self.ids.textID_signprofmail.text = ""
        self.ids.textID_signprofpass.text = ""


class Signed(Screen):
    print("INITIALIZED: Sign-Up Sucess SCREEN")


class MyRooms(Screen):
    print("INITIALIZED: MyRooms SCREEN")

    def clearList(self):
        self.ids.buttonID_CreateSummary.disabled = False
        self.ids.listID_MainList.clear_widgets()

    def createSummary(self):
        print("CREATING SUMMARY")

    def reloadRooms(self):
        print("RELOADING ROOM LIST")
        global sm
        sm.get_screen("kv_LoggedProf").loadRooms()

    def setLenient(self):
        print("DEBUG: SET THRESHOLD TO LENIENT")
        user.setThreshold("LENIENT")
        self.ids.buttonID_Lenient.opacity = 1
        self.ids.buttonID_Moderate.opacity = .5
        self.ids.buttonID_Strict.opacity = .5
    
    def setModerate(self):
        print("DEBUG: SET THRESHOLD TO MODERATE")
        user.setThreshold("MODERATE")
        self.ids.buttonID_Lenient.opacity = .5
        self.ids.buttonID_Moderate.opacity = 1
        self.ids.buttonID_Strict.opacity = .5
    
    def setStrict(self):
        print("DEBUG: SET THRESHOLD TO STRICT")
        user.setThreshold("STRICT")
        self.ids.buttonID_Lenient.opacity = .5
        self.ids.buttonID_Moderate.opacity = .5
        self.ids.buttonID_Strict.opacity = 1


class MySummary(Screen):
    print("INITIALIZED: MySummary SCREEN")



class MyData(Screen):
    def showGraph(self):
        list_actions = user.getResults()
        x = []
        height = []
        color_bar = []
        for action in list_actions:
            actions_split = action.split("]")
            if actions_split[0] == "Right_Head_Tilt":
                x.append(actions_split[0])
                height.append(int(actions_split[1]))
                color_bar.append("red")
            elif actions_split[0] == "Left_Head_Tilt":
                x.append(actions_split[0])
                height.append(int(actions_split[1]))
                color_bar.append("green")
            elif actions_split[0] == "Down_Head_Tilt":
                x.append(actions_split[0])
                height.append(int(actions_split[1]))
                color_bar.append("orange")
            elif actions_split[0] == "Centered":
                x.append(actions_split[0])
                height.append(int(actions_split[1]))
                color_bar.append("blue")
            else:
                print("ACTION NOT FOUND: ", actions_split)
        plt.bar(x, height, width = 0.5, color=color_bar)
        plt.xlabel("Action")
        plt.ylabel("Diration in Seconds")
        plt.title("GRAPH")
        plt.show()
            

    def clearList(self):
        self.ids.imageID_DataImage.source = "UI/default.jpg"
        self.ids.listID_DataList.clear_widgets()

# Main build class
class OECD(MDApp):
    def build(self):
        global sm
        self.load_kv('main.kv')
        sm = ScreenManager()
        sm.add_widget(LoggedProfessor(name = 'kv_LoggedProf'))
        sm.add_widget(LoginScreen(name = 'kv_login'))
        sm.add_widget(MyRooms(name = 'kv_MyRooms'))
        sm.add_widget(MyData(name = 'kv_MyData'))
        sm.add_widget(MySummary(name = 'kv_MySummary'))
        sm.add_widget(MyRooms(name = 'kv_MyRooms'))
        sm.add_widget(MainStudent(name = 'kv_MainStudent'))
        sm.add_widget(MainStudent(name = 'kv_MainStudent'))
        sm.add_widget(MainProfessor(name = 'kv_MainProf'))
        sm.add_widget(SignProfessor(name = 'kv_SignProf'))
        sm.add_widget(Signed(name = 'kv_Signed'))
        sm.add_widget(LoggedStudent(name = 'kv_LoggedStudent'))
        sm.add_widget(MainAdmin(name = 'kv_MainAdmin'))
        print("INITIALIZED: SCREEN MANAGER AND SCREENS")
        return sm


if __name__ == "__main__":
    print("INITIALIZED: MAIN")
    db = databaseInit()
    user = userInit()
    # tracking = visualTracking()
    Window.size = (600, 300)
    OECD().run()

