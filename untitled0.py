from keras.models import load_model
import numpy as np
from PIL import Image
import os

# Cargar modelo pre-entrenado de FaceNet
model = load_model('facenet_keras.h5')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 160))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return img

def get_embedding(model, image_path):
    img = preprocess_image(image_path)
    embedding = model.predict(img)
    return embedding[0]

# Generar embeddings para las imágenes de la familia
family_embeddings = {}
family_directory = "C:/Users/kikhe/Documents/GitHub/FaceNet/members-photo"
for person_name in os.listdir(family_directory):
    person_images_names = os.listdir(os.path.join(family_directory, person_name))
    embeddings = []
    for image_name in person_images_names:
        image_path = os.path.join(family_directory, person_name, image_name)
        embedding = get_embedding(model, image_path)
        embeddings.append(embedding)
    family_embeddings[person_name] = np.mean(embeddings, axis=0)

