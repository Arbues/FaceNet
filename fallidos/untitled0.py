import tensorflow as tf
import numpy as np
from PIL import Image

# Desactivar la ejecución ansiosa
tf.compat.v1.disable_eager_execution()

# Cargar el grafo y restaurar los pesos del checkpoint
def load_model(model_dir):
    # Reiniciar el grafo
    tf.compat.v1.reset_default_graph()

    # Cargar el grafo desde el archivo .meta
    saver = tf.compat.v1.train.import_meta_graph(f'{model_dir}/model-20180408-102900.meta')

    # Iniciar sesión y restaurar los pesos desde el checkpoint
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        return sess, tf.compat.v1.get_default_graph()

# Preprocesar la imagen
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 160))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return img

# Obtener el embedding para una imagen
def get_embedding(sess, graph, image_path):
    images_placeholder = graph.get_tensor_by_name('input:0')
    embeddings = graph.get_tensor_by_name('embeddings:0')
    phase_train_placeholder = graph.get_tensor_by_name('phase_train:0')

    img = preprocess_image(image_path)
    feed_dict = {images_placeholder: img, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding[0]

# Ruta al directorio del modelo
model_dir = 'C:/Users/kikhe/Documents/GitHub/FaceNet/modelo/20180408-102900'

# Cargar el modelo
sess, graph = load_model(model_dir)

# Obtener el embedding para una imagen específica
image_path = 'path_to_your_image.jpg'
embedding = get_embedding(sess, graph, image_path)
print("Embedding obtenido:", embedding)
