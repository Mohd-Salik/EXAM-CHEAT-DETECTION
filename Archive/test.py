from pickle import TRUE
import numpy as np
import os
import time
from datetime import datetime

images = ['2022-03-14-01-17-36-766937]qwe.jpg', '2022-03-14-01-17-37-925808]qwe.jpg', '2022-03-14-01-17-44-533801]qwe.jpg', '2022-03-14-01-17-53-719882]qwe.jpg', '2022-03-14-01-18-06-345401]qwe.jpg']
actions =  ['Centered', 'Right_Head_Tilt', 'Centered', 'Down_Head_Tilt', 'Centered']
time_slot = ['1', '6', '9', '12']

for index in range(len(images)):
    print(index)
    action = "{}]{}".format(actions[index], time_slot[index])
    
    data = {"Action": action, "File" : images[index]}
    print("APPENDING: ", data)
