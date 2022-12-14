
from importlib.resources import path

import os

import numpy as np
from PIL import Image
import pickle
import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR, "images")
current_id = 0
labels_ids = {}
y_labels = []
x_train = []
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

reconizer = cv2.face.LBPHFaceRecognizer_create() 

for root, dirs, files in os.walk(image_dir):
  for file in files:
    if file.endswith("png") or file.endswith("jpg"):
      path = os.path.join(root,file)
      label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
    

      
      if(label in labels_ids):
        pass
      else:
        labels_ids[label] = current_id
        current_id += 1
      id_ = labels_ids[label]
      pil_image = Image.open(path).convert("L")
      image_array = np.array(pil_image, "uint8")
      faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
      
      for(x,y,w,h) in faces:
        roi = image_array[y:y+h, x:x+w]
        x_train.append(roi)
        y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
  pickle.dump(labels_ids, f)

reconizer.train(x_train, np.array(y_labels))
reconizer.save("trainner.yml")