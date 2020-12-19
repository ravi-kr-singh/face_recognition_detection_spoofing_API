def btp1(image):
    
  import cv2
  import numpy as np
  from tensorflow.keras.models import load_model
  from tensorflow.keras.preprocessing.image import img_to_array
  from keras import backend as k
  import pickle
  import face_recognition
  import csv
  import time
  
  resolution=360
  pred,pred1=0,0
  names=[]
  model=load_model("livelinessmodel.h5", compile=False)
  model1=load_model("livelinessedgemodel.h5", compile=False)
  cascade = "haarcascade_frontalface_default.xml"   
  data = pickle.loads(open("data.csv", "rb").read())
  
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  boxes = face_recognition.face_locations(rgb, model='hog')
  if(len(boxes)!=1): return 0
  encodings = face_recognition.face_encodings(rgb, boxes)
  matches = face_recognition.compare_faces(data["encodings"],encodings[0])
  names = "Unknown"
  if True in matches:
      matchedIdxs = [i for (i, b) in enumerate(matches) if b]
      counts = {}
      for i in matchedIdxs:
          name = data["names"][i]
          counts[name] = counts.get(name, 0) + 1
      names = max(counts, key=counts.get)
    
  (h, w) = image.shape[:2]
  if(h>w): image=cv2.resize(image,(resolution,round(h*resolution/w)))
  else: image=cv2.resize(image,(round(w*resolution/h),resolution))
  (h,w)=image.shape[:2]

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faceCascade = cv2.CascadeClassifier(cascade)
  face = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=7,
    minSize=(round(h/3.5), round(w/3.5))
  )
  if(len(face)!=1): return(0)
  (xf,yf,wf,hf)=face[0] 
  
  face=cv2.resize(image[yf:yf+wf, xf:xf+hf],(32,32))
  face=face.astype("float")/255
  face=img_to_array(face)
  face=np.expand_dims(face,axis=0)
  
  pred=model.predict(face)
  
  sigma=0.33
  gray=cv2.resize(gray,(100,100))
  blurred = cv2.GaussianBlur(gray, (3, 3), 0)
  v = np.median(blurred)
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(blurred, lower, upper)
  edged=edged.astype(int)
  edged=np.stack((edged,)*3, axis=-1)
  edged=edged.astype("int")/255
  edged=img_to_array(edged)
  edged=np.expand_dims(edged,axis=0)
  pred1=model1.predict(edged)
  k.clear_session()
  

  
  path1='G:/btp1/fail/'
  path2='G:/btp1/success/'
  #return(pred[0][1],pred1[0][1])
  if(pred[0][1]>0.95): 
      if(names!='Unknown'): flag=1
      else: return(names)
  
  elif(pred1[0][1]<0.1): 
      if(names!='Unknown'): flag=0
      else: return 0
  
  elif(pred[0][1]>=0.6 or pred1[0][1]>=0.6): 
      if(names!='Unknown'): flag=1
      else: return(names)
  
  else: 
      if(names!='Unknown'): flag=0
      else: return 0
  
  if flag==1:  
          data= [names,str(time.strftime("%b_%m-%H_%M")),'success']
          with open('record.csv', 'a') as work:
              z= csv.writer(work)
              z.writerow(data)
          filename=names+'__'+str(time.strftime("%b_%m-%H_%M"))+'.jpg'
          cv2.imwrite(path2+filename,image)
          return(names)
  else:   
          data= [names,str(time.strftime("%b_%m-%H_%M")),'fail']
          with open('record.csv', 'a') as work:
              z= csv.writer(work)
              z.writerow(data)
          filename=names+'__'+str(time.strftime("%b_%m-%H_%M"))+'.jpg'
          cv2.imwrite(path1+filename,image)
          return(0)