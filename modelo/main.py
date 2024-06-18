#%% IMPORT
import time
import cv2 as cv
import numpy as np
import os
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import register_adapter

def adapt_int32(value):
  return int(value)

register_adapter(np.int32, adapt_int32)
register_adapter(np.int64, adapt_int32)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "mydatabase")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")
def insert_detection(person_id, face_array):
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
    )
    cur = conn.cursor()
    insert_query = f"""
    INSERT INTO face_events(x, y, w, h, person_id) VALUES ({face_array[0]}, {face_array[1]}, {face_array[2]}, {face_array[3]}, '{person_id}')
    """
    cur.execute(insert_query)
    conn.commit()
    cur.close()
    conn.close()

#%% INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
EMBEDDED_X = faces_embeddings["EMBEDDED_X"]
Y_encoded = faces_embeddings["Y"]
encoder = LabelEncoder()
encoder.fit(Y_encoded)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("kmeans_model_160x160.pkl", 'rb'))

#%% WHILE LOOP
# ip_camera_url = os.getenv('CAMERA_URL')
ip_camera_url = "rtsp://acecom:1964@192.168.1.19:8080/h264_ulaw.sdp"
while True:
    cap = cv.VideoCapture(ip_camera_url)
    # cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream or file, trying reconnect...")
        time.sleep(3)
    else:
        break
detected_ids = {}
DETECTION_TIME = 3
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara")

else:
    # cv.namedWindow("Face Recognition")
    while True:
        # if cv.getWindowProperty('Face Recognition', 0) < 0:
        #     # Check if the window was closed
        #     break
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara")
            break

        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        # print(faces)
        for (x, y, w, h) in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            final_name = encoder.inverse_transform(face_name)[0]
            # print(final_name)

            current_time = time.time()

            if final_name not in detected_ids:
                detected_ids[final_name] = current_time

            if current_time - detected_ids[final_name] >= DETECTION_TIME:
                print("INSERCCION BASE DE DATOS!!!")
                insert_detection(final_name, [x, y, w, h])
                del detected_ids[final_name]

            for key in [key for key, timestamp in detected_ids.items()
                        if current_time-timestamp > DETECTION_TIME]:
                del detected_ids[key]
                # print("Borrando por inactividad: ", key)
            # cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            # cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)


        # cv.imshow("Face Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # cv.destroyWindow("Face Recognition")

#%% Check if the window needs to be destroyed again (in case of any error)
# if cv.getWindowProperty('Face Recognition', cv.WND_PROP_VISIBLE) >= 1:
#     cv.destroyAllWindows()
