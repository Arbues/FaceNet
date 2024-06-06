import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import pickle

# Cargar la base de datos de embeddings
with open('embeddings_database.pkl', 'rb') as f:
    embeddings_database = pickle.load(f)

# Cargar el modelo FaceNet
MODEL_DIR = r'C:\Users\kikhe\Documents\GitHub\FaceNet\modelo\20180408-102900'  # Reemplaza con la ruta a tu directorio de modelo
MODEL_PATH = os.path.join(MODEL_DIR, '20180408-102900.pb')
META_FILE = os.path.join(MODEL_DIR, 'model-20180408-102900.meta')
CKPT_FILE = os.path.join(MODEL_DIR, 'model-20180408-102900.ckpt-90')

def load_model(model_path):
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

def preprocess_image(image):
    img = cv2.resize(image, (160, 160))
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return img

# Función para obtener el embedding de una imagen
def get_embedding(image, session, images_placeholder, embeddings, phase_train_placeholder):
    img = preprocess_image(image)
    feed_dict = {images_placeholder: img[np.newaxis, ...], phase_train_placeholder: False}
    embedding = session.run(embeddings, feed_dict=feed_dict)
    return embedding

# Cargar el modelo
load_model(MODEL_PATH)

# Restaurar los pesos del modelo
sess = tf.Session()
saver = tf.train.import_meta_graph(META_FILE)
saver.restore(sess, CKPT_FILE)

# Obtener referencias a los placeholders y las operaciones necesarias
graph = tf.get_default_graph()
images_placeholder = graph.get_tensor_by_name("input:0")
embeddings = graph.get_tensor_by_name("embeddings:0")
phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar caras en el fotograma
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Recortar la cara
        face_img = frame[y:y+h, x:x+w]

        # Obtener el embedding de la cara
        embedding = get_embedding(face_img, sess, images_placeholder, embeddings, phase_train_placeholder)

        # Comparar el embedding con la base de datos y encontrar la persona más cercana
        min_distance = float('inf')
        recognized_person = None
        for person_name, person_embedding in embeddings_database.items():
            distance = np.linalg.norm(embedding - person_embedding)
            if distance < min_distance:
                min_distance = distance
                recognized_person = person_name

        # Dibujar un cuadro alrededor de la cara y mostrar el nombre de la persona
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el fotograma resultante
    cv2.imshow('Face Recognition', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
