import pyrebase
import urllib

firebaseConfig = {
    'apiKey': "AIzaSyDP7ELddk4JrFfNZJtXcsCgNwrRow_Jw2E",
    'authDomain': "oecd-90c43.firebaseapp.com",
    'projectId': "oecd-90c43",
    'storageBucket': "oecd-90c43.appspot.com",
    'messagingSenderId': "466972383779",
    'appId': "1:466972383779:web:5de25fd2b26c719549ef96",
    'measurementId': "G-X8HMETBY30",
    "databaseURL" : "https://oecd-90c43-default-rtdb.asia-southeast1.firebasedatabase.app/"
}

firebase = pyrebase.initialize_app(firebaseConfig)

database = firebase.database()
auth = firebase.auth()
storage = firebase.storage()

# people=database.child("ROOMS").child("test_professor@gmail+com").get()
# for person in people.each():
    # if person.val()['room name']=='jande':
    # print(person.val()['room name'])
# studs = [
#     2018100403,
#     2018104005,
#     2018105506,
#     2019100526,
#     2019100950,
#     2019102315,
#     2019103706,
#     2020118601,
#     2020140149,
#     2020147191,
#     2021140723,
#     2021140753,
#     2021141002,
#     2021141018,
#     2021141069,
#     2021141132,
#     2021150051,
#     2021150819,
#     2021160070,
#     2021160709]

# entrya121 = [
#     2018104005,
#     2018107004,
#     2019100950,
#     2019102233,
#     2020140149,
#     2020190014,
#     2021150051,
#     2021241203]

# baste = [
#     "ALBINO",
#     "AVILA",
#     "Cabahug",
#     "Caluyong",
#     "LAURENTE",
#     "MACAHINE",
#     "Magalona",
#     "Montano",
#     "Oncada",
#     "Ortaliz",
#     "Panggoy",
#     "Pineda",
#     "Sia",
#     "Stones",
#     "Tomas"]

# for entry in baste:
#     print("-------", entry)
#     student = database.child("ROOMS").child("test_professor@gmail+com").child("SIR_BASTE").child(str(entry)).get()
#     total_left = []
#     total_right = []
#     total_center = []
#     total_lower = []
#     for x in student.each():
#         try:
#             action = x.val()["Action"]
#             action = action.split("]")
#         except:
#             print("TIME")
#         if action[0] == "Right_Head_Tilt":
#             total_right.append(int(action[1]))
#         elif action[0] == "Left_Head_Tilt":
#             total_left.append(int(action[1]))
#         elif action[0] == "Down_Head_Tilt":
#             total_lower.append(int(action[1]))
#         elif action[0] == "Centered":
#             total_center.append(int(action[1]))
#         else:
#             print("ACTION NOT FOUND: ", action[0])
#     # print(total_center)
#     # print(total_right)
#     # print(total_lower)
#     # print(sum(total_left))

#     summation = len(total_center) + len(total_right) + len(total_lower) + len(total_left)

#     center_pers = (sum(total_center)/summation)*100
#     left_pers = (sum(total_left)/summation)*100
#     right_pers = (sum(total_right)/summation)*100
#     lower_pers = (sum(total_lower)/summation)*100

#     # print("Centered: ", len(total_center), "\t%: ", round(center_pers, 2),
#     #     "\nLeft Head Tilt: ", len(total_left), "\t%: ", round(left_pers, 2),
#     #     "\nRight Head Tilt: ", len(total_right), "\t%: ", round(right_pers, 2),
#     #     "\nLower Head Tilt: ", len(total_lower), "\t%: ", round(lower_pers, 2),
#     #     "\nTOTAL: ", summation
#     # )

#     print("Centered ", "%: ", round(center_pers, 2),
#         "\nLeft Head Tilt ", "%: ", round(left_pers, 2),
#         "\nRight Head Tilt ", "%: ", round(right_pers, 2),
#         "\nLower Head Tilt ", "%: ", round(lower_pers, 2),
#         "\nTOTAL: ", summation
#     )


# AUTHENTICATION
# email = input ("email: ")
# password = input("pass: ")
try:   
    auth.sign_in_with_email_and_password("test_professor@gmail.com", "test_professor")
    print("sucess")
except:
    print("INVALID USER OR PASWORD") 

# email = input ("sign up email: ")
# password = input("sign up pass: ")
# confirm = input("confirm password: ")
# if password==confirm:
# try: 
#     auth.create_user_with_email_and_password("greatefreat@gmail.com", "leroynJENKINS123")
#     print("sucess")
# except:
#     print("Email already exist")

# STORAGE
# filename=input("ENTER name of file: ")
# cloudefilename=input("ENTNTER FILE ON THE CLOUD: ")
# storage.child(cloudefilename).put(filename)
# print(storage.child(cloudefilename).get_url(None))

# DOWNLOAD
# cloudefilename=input("ENTNTER FILE ON THE CLOUD: ")
# storage.child("test/default.jpg").download("", "preview.jpg")
# storage.child("ROOMS/test_professor@gmail com/default.jpg").download("", "test1.jpg")
# READING TXT/FILE
# cloudfilename=input("INPUT FILE NAME IN CLOUD:")
# url=storage.child("image1.jfif").get_url(None)
# f=urllib.request.urlopen(url).read()
# print(f)

# DATABASE
#INSERTING DATA
# data={'age': 40, 'address':'USA', 'employed': False, 'name': 'oracle smith'}
# database.push(data)
# database.child("CS0034EXAMINATION").set(data)
# database.child("people").child("myownid").set(data)
# UPDATE DATA
# database.child("people").child("myownid").update({'name':'jande'})
# database.child("people").child("myownid").update({'newdata':'wowex'})

#UPDATE WITH NO ID
# people=database.child("people").get()
# for person in people.each():
#     print(person.val())
#     print(person.key())

# rooms=
# database.child("ROOMS").child("qwe@gmail+com").get()
# for owner in rooms.each():
#     print(owner.key())

# data = {"Action": "Lookleft]8", "File": "2022-03-11 08:53:44.496813]201812"}

# database.child("ROOMS").child("qwe@gmail+com").child("CS004-Examinations").child("201812").push(data)

# people = database.child("ROOMS").child("test_professor@gmail+com").child("MAM_CHERRY_A121").child("2018100403").get()

# for person in people.each():
#     print(person.val()["Action"])
#     print(person.val()["File"])

    # print(person.key())
    # if person.val()['name']=='jande':
    #     database.child('people').child(person.key()).update({'age': 22})

# DELETE DATA
# for person in people.each():
#     if person.val()['name']=='jande':
#         database.child("people").child("person").child("age").remove()

# READ
# people=database.child("people").child('-MxVBxDrL477I7nzoQaT').get()
# print(people.val()['age'] + 1)


# GET ALL JANE
# people=database.child("people").order_by_child("name").equal_to("jande").get()

# for person in people.each():
#     print(person.val()['age'])