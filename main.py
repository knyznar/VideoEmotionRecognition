import cv2
import sys
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import model
model = load_model('model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop_frame = frame[y:y + h, x:x + w]
        gray_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
        # break

        gray_frame_resized = np.resize(gray_frame, (1, 48, 48))  # cos nie tak z reshapem albo nwm, nie umie w Happy
        gray_frame_resized = np.expand_dims(gray_frame_resized, -1)

        Y_predicted = model.predict(gray_frame_resized)
        predicted_emotion = emotions[np.argmax(Y_predicted[0])]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
