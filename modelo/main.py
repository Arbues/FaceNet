#%% IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

#%% INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("C:/Users/kikhe/Documents/GitHub/FaceNet/modelo/faces_embeddings_done_4classes.npz")
EMBEDDED_X = faces_embeddings["EMBEDDED_X"]
Y_encoded = faces_embeddings["Y"]
encoder = LabelEncoder()
encoder.fit(Y_encoded)
haarcascade = cv.CascadeClassifier("C:/Users/kikhe/Documents/GitHub/FaceNet/modelo/haarcascade_frontalface_default.xml")
model = pickle.load(open("C:/Users/kikhe/Documents/GitHub/FaceNet/modelo/svm_model_160x160.pkl", 'rb'))


#%% WHILE LOOP
cap = cv.VideoCapture(0)  # Cambia el índice a 0

if not cap.isOpened():
    print("Error: No se puede acceder a la cámara")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara")
            break
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        for x, y, w, h in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv.resize(img, (160,160))  # 1x160x160x3
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            final_name = encoder.inverse_transform(face_name)[0]
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
            cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("Face Recognition:", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Cambié la condición para salir
            break

    cap.release()
    cv.destroyAllWindows()

#%% Verificación de cv.destroyAllWindows
if cv.getWindowProperty('Face Recognition:', cv.WND_PROP_VISIBLE) >= 1:
    cv.destroyAllWindows()

