
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow.compat.v1 as tf
import numpy as np
import os
from PIL import Image

# Ruta al modelo pre-entrenado de FaceNet
MODEL_DIR = r'C:\Users\kikhe\Documents\GitHub\FaceNet\modelo\20180408-102900'  # Reemplaza con la ruta a tu directorio de modelo
MODEL_PATH = os.path.join(MODEL_DIR, '20180408-102900.pb')
META_FILE = os.path.join(MODEL_DIR, 'model-20180408-102900.meta')
CKPT_FILE = os.path.join(MODEL_DIR, 'model-20180408-102900.ckpt-90')

# Función para cargar el modelo FaceNet
def load_model(model_path):
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

# Función para preprocesar las imágenes
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 160))  # FaceNet espera imágenes de 160x160
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return img

# Función para obtener el embedding de una imagen
def get_embedding(image_path, session, images_placeholder, embeddings, phase_train_placeholder):
    img = preprocess_image(image_path)
    feed_dict = {images_placeholder: img, phase_train_placeholder: False}
    embedding = session.run(embeddings, feed_dict=feed_dict)
    return embedding[0]

# Ruta al directorio que contiene las imágenes de las personas
PEOPLE_DIR = "C:/Users/kikhe/Documents/GitHub/FaceNet/members-photo"  # Reemplaza con la ruta a tu directorio de imágenes

# Crear una base de datos de embeddings
def create_embeddings_database(people_dir, model_path, meta_file, ckpt_file):
    embeddings_database = {}

    with tf.Session() as sess:
        # Cargar el modelo
        load_model(model_path)

        # Restaurar los pesos del modelo
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, ckpt_file)

        # Obtener referencias a los placeholders y las operaciones necesarias
        graph = tf.get_default_graph()
        images_placeholder = graph.get_tensor_by_name("input:0")
        embeddings = graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

        for person_name in os.listdir(people_dir):
            person_dir = os.path.join(people_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            person_embeddings = []
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                embedding = get_embedding(image_path, sess, images_placeholder, embeddings, phase_train_placeholder)
                person_embeddings.append(embedding)

            # Promediar los embeddings de cada persona para obtener una representación única
            embeddings_database[person_name] = np.mean(person_embeddings, axis=0)

    return embeddings_database

# Crear la base de datos de embeddings
embeddings_database = create_embeddings_database(PEOPLE_DIR, MODEL_PATH, META_FILE, CKPT_FILE)

# Guardar la base de datos de embeddings en un archivo para uso futuro
import pickle
with open('embeddings_database.pkl', 'wb') as f:
    pickle.dump(embeddings_database, f)

print("Base de datos de embeddings creada y guardada exitosamente.")

