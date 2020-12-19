# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 00:08:40 2020

@author: naman
"""

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

resolution=360
knownEncodings = []
knownNames = []
cascade = "haarcascade_frontalface_default.xml"   
faceCascade = cv2.CascadeClassifier(cascade)
path = 'G:/btp1/real'
i=0
for filename in os.listdir(path):
    i+=1
    print("Processing image ",i,"/",len(os.listdir(path)))
    name=filename[:-4]
    image = cv2.imread(path+'/'+filename)
    (h, w) = image.shape[:2]
    if(h>w): image=cv2.resize(image,(resolution,round(h*resolution/w)))
    else: image=cv2.resize(image,(round(w*resolution/h),resolution))
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    #(xf,yf,wf,hf)=boxes[0]   
    #face= rgb[xf:wf, hf:yf]
    #cv2.imshow(filename,face)
    if len(boxes)!=1: continue
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
        
data = {"encodings": knownEncodings, "names": knownNames}
f = open('data.csv', "wb")
f.write(pickle.dumps(data))
f.close()        