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
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

for x in range(0,10):
    DATA_PATH = os.path.join('FINAL_TRAINING_NO_IRIS') # Path for exported data, numpy arrays
    actions = np.array(['Left_Head_Tilt',
        'Up_Head_Tilt',
        'Down_Head_Tilt',
        'Right_Head_Tilt',
        'Centered']) 
    no_sequences = 30 # Thirty videos worth of data
    sequence_length = 30 # Videos are going to be 30 frames in length


    label_map = {label:num for num, label in enumerate(actions)}

    # print(label_map)
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    # print("SEQUENCE: ", np.array(sequences).shape)
    # print("LABEL: ", np.array(labels).shape)
    X = np.array(sequences)
    print(X.shape)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # # CREATING MODEL
    log_dir = os.path.join('MODEL LOGS')
    tb_callback = TensorBoard(log_dir=log_dir)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1692)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])


    history = model.fit(X_train, y_train, validation_split=0.3, epochs=200, verbose=0, callbacks=[tb_callback])

    # evaluate the model
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)

    print("\nMMODEL {}:".format(x), accuracy, loss, f1_score, precision, recall)