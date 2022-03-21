# DEPENDENCIES
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

# DATA_PATH = os.path.join('FINAL_TRAINING_NO_IRIS') # Path for exported data, numpy arrays
actions = np.array(['Left_Head_Tilt',
    'Up_Head_Tilt',
    'Down_Head_Tilt',
    'Right_Head_Tilt',
    'Centered']) 
# no_sequences = 30 # Thirty videos worth of data
# sequence_length = 30 # Videos are going to be 30 frames in length


# label_map = {label:num for num, label in enumerate(actions)}

# print(label_map)
# sequences, labels = [], []
# for action in actions:
#     for sequence in range(no_sequences):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])

# print("SEQUENCE: ", np.array(sequences).shape)
# print("LABEL: ", np.array(labels).shape)
# X = np.array(sequences)
# print(X.shape)
# y = to_categorical(labels).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# # # CREATING MODEL
# log_dir = os.path.join('MODEL LOGS')
# tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1692)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('HeadV2_300_epoch.h5')
model.summary()

