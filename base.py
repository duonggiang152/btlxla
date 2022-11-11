import numpy as np
import cv2
import pickle
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
reconizer = cv2.face.LBPHFaceRecognizer_create() 
reconizer.read("trainner.yml")
labels = {"giang": 0}
with open("labels.pickle", 'rb') as f:
  og_labels = pickle.load(f)
  labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)

while(True):
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
  
  for( x, y, w, h) in faces:
    print(x,y,w,h)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x: x + w]
    
    id_, conf = reconizer.predict(roi_gray)
    
    if(conf >= 45 and conf <= 85):
      print(id_)
      print(labels[id_])
    
    img_item = 'my-image.png'
    cv2.imwrite(img_item, roi_gray)
    
    color = (255, 0, 0) #BGR 0-255
    stroke = 2
    
    end_cor_width = x + w
    end_cor_height = y + h
    
    cv2.rectangle(frame, (x,y), (end_cor_width, end_cor_height), color, stroke)
    
    
  cv2.imshow('frame', frame)
  if cv2.waitKey(20) & 0xFF == ord('q'):
    break
  
cap.release()
cv2.destryAllWindows()