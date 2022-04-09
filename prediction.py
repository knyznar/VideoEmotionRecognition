import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("TensorFlow version: " + tf.__version__)
print("Keras version: " + keras.__version__)
print("Numpy version: " + np.__version__)


def translate_emotion_value(value): #function used for visualization purposes only
  if(value==0):
    return 'Angry'
  elif(value==1):
    return 'Disgust'
  elif(value==2):
    return 'Fear'
  elif(value==3):
    return 'Happy'
  elif(value==4):
    return 'Sad'
  elif(value==5):
    return 'Surprise'
  elif(value==6):
    return 'Neutral'


model = load_model('model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

image = Image.open('0.jpg').resize((48, 48), Image.ANTIALIAS)
data = [np.array(image)]
image_reshaped = np.resize(image, (1, 48, 48))
image_reshaped = np.expand_dims(image_reshaped, -1)
print(image_reshaped.shape)
print(type(image_reshaped))

Y_predicted = model.predict(image_reshaped)
print(translate_emotion_value(np.argmax(Y_predicted)))
print(Y_predicted)


