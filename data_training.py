import os  
import numpy as np  
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

is_init = False
label = []
dictionary = {}
c = 0


for file in os.listdir():
    if file.endswith(".npy") and file != "labels.npy":  
        data = np.load(file)

        if data.ndim == 1:
            data = data.reshape(-1, data.shape[0])  # Reshape to (samples, features)

        if not is_init:
            is_init = True 
            X = data
            size = X.shape[0]
            y = np.array([file.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([file.split('.')[0]] * size).reshape(-1, 1)))

        label.append(file.split('.')[0])
        dictionary[file.split('.')[0]] = c  
        c += 1


y = np.array([dictionary[val[0]] for val in y], dtype="int32")

y = to_categorical(y)


indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

input_shape = (X.shape[1],)  


ip = Input(shape=input_shape)
m = Dense(128, activation="tanh")(ip)
m = Dense(64, activation="tanh")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(X, y, epochs=80)


model.save("model.h5")
np.save("labels.npy", np.array(label))