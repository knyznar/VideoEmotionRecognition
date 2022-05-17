import cv2
import sys
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import model
model = load_model('model_keras_tuner.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

videoCapture = cv2.VideoCapture(0)

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
frameCounter = 0

while True:
    frameCounter += 1
    if frameCounter % 3 != 0:
        continue

    # Capture frame-by-frame
    ret, frame = videoCapture.read()

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesList = faceCascade.detectMultiScale(
        grayFrame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in facesList:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropFrame = grayFrame[y:y + h, x:x + w]


        cropFrameResized = np.resize(cropFrame, (1, 48, 48))
        cropFrameResized = np.expand_dims(cropFrameResized, -1)

        Y_predicted = model.predict(cropFrameResized)
        predicted_emotion = emotions[np.argmax(Y_predicted[0])]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
videoCapture.release()
cv2.destroyAllWindows()
