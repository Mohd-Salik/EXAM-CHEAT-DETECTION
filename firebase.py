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
# auth = firebase.auth()
# storage = firebase.storage()

# AUTHENTICATION
# email = input ("email: ")
# password = input("pass: ")
# try:   
#     auth.sign_in_with_email_and_password(email, password)
#     print("sucess")
# except:
#     print("INVALID USER OR PASWORD") 

# email = input ("sign up email: ")
# password = input("sign up pass: ")
# confirm = input("confirm password: ")
# if password==confirm:
#     try: 
#         auth.create_user_with_email_and_password(email, password)
#         print("sucess")
#     except:
#         print("Email already exist")

# STORAGE
# filename=input("ENTER name of file: ")
# cloudefilename=input("ENTNTER FILE ON THE CLOUD: ")
# storage.child(cloudefilename).put(filename)
# print(storage.child(cloudefilename).get_url(None))

# DOWNLOAD
# cloudefilename=input("ENTNTER FILE ON THE CLOUD: ")
# storage.child(cloudefilename).download("", "image1.jfif")

# READING TXT/FILE
# cloudfilename=input("INPUT FILE NAME IN CLOUD:")
# url=storage.child(cloudfilename).get_url(None)
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
database.child("ROOMS").child("qwe@gmail+com").get()
# for owner in rooms.each():
#     print(owner.key())

data = {"Action": "Lookleft8", "File": "2022-03-11 08:53:44.496813]201812"}

people = database.child("ROOMS").child("qwe@gmail+com").child("CS004-Examinations").child("201812").get()

for person in people.each():
    if person.key() == "-MxtOARqIIPt8QDFBZ23":
        print(person.val())
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