
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


def buildmodel(width, height, depth, classes):

    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if (K.image_data_format() == "channels_first"):
	        inputShape = (depth, height, width)
	        chanDim = 1
 
    model.add(Conv2D(16, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
	 
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model


cascade = "haarcascade_frontalface_default.xml"   
faceCascade = cv2.CascadeClassifier(cascade)
INIT_LR = 0.0001
BS = 10
EPOCHS = 30
resolution=240
data = []
labels = []
sigma=0.33

path = 'G:/liveliness/fake'
for filename in os.listdir(path):
 with open(os.path.join(path, filename), 'r') as f: 
    path1= path+'/'+filename
    image = cv2.imread(path1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h,w)=image.shape[:2]
    face = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=7,
    minSize=(round(h/3.5), round(w/3.5))
    )
    if(len(face)!=1): continue
    gray=cv2.resize(gray,(50,50))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    edged=edged.astype(int)
    edged=np.stack((edged,)*3, axis=-1)
    data.append(edged)
    labels.append(0)

path = 'G:/liveliness/real'
for filename in os.listdir(path):
 with open(os.path.join(path, filename), 'r') as f: 
    path1= path+'/'+filename
    image = cv2.imread(path1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h,w)=image.shape[:2]
    face = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=7,
    minSize=(round(h/3.5), round(w/3.5))
    )
    if(len(face)!=1): continue
    gray=cv2.resize(gray,(50,50))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    edged=edged.astype(int)
    edged=np.stack((edged,)*3, axis=-1)
    data.append(edged)
    labels.append(0)

data=np.stack(data,axis=0)
data = np.array(data, dtype="int") / 255  
labels=to_categorical(labels,2) 
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2)

print("Training model")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = buildmodel(width=50, height=50,depth=3, classes=2)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
          
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size= BS, epochs=EPOCHS)   

predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))       
model.save('livelinessedgemodel.h5')