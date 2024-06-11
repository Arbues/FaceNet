from mtcnn.mtcnn import MTCNN
import cv2 as cv
import os
from keras_facenet import FaceNet
import numpy as np


#SVM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#%%
class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()


    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr


    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory +'/'+ sub_dir+'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)


    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')

#%% 
# Instanciación de la clase FACELOADING y carga de datos
faceloading = FACELOADING("C:/Users/kikhe/Documents/GitHub/FaceNet/members-photo")
# en X estan las car
X, Y = faceloading.load_classes()

#%%
# Codificación de las etiquetas de las clases
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
#%%
embedder = FaceNet()

def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

#%%
# Obtención de embeddings
EMBEDDED_X = [get_embedding(face) for face in X]
EMBEDDED_X = np.asarray(EMBEDDED_X)

#%%
# Guardar los embeddings en un archivo comprimido
file_path = r'C:\Users\kikhe\Documents\GitHub\FaceNet\modelo\faces_embeddings_done_4classes.npz'

# Guardar el archivo npz comprimido en la ruta especificada
np.savez_compressed(file_path, EMBEDDED_X=EMBEDDED_X, Y=Y_encoded)

print(f"Archivo npz guardado correctamente en {file_path}")

#%% SVM
X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y_encoded, shuffle=True, random_state=17)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)
#accuracy
accuracy_score(Y_train, ypreds_train)
accuracy_score(Y_test,ypreds_test)  
#%%
# Clasificación utilizando K-means ajustada
k = len(np.unique(Y_encoded))  # El número de clusters es igual al número de personas únicas
kmeans = KMeans(n_clusters=k, n_init=10, random_state=17)
kmeans.fit(EMBEDDED_X)

#%%
# Predicciones y evaluación del modelo (Opcional, depende de cómo se quiera evaluar K-means)
# Aquí se podría calcular la inercia o algún otro indicador de la calidad del clustering:
print("Inercia del modelo K-means:", kmeans.inertia_)

#%%
# Guardar el modelo K-means
model_path = r'C:\Users\kikhe\Documents\GitHub\FaceNet\modelo\kmeans_model_160x160.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(kmeans, f)
#%%
#save the model
model_path = r'C:\Users\kikhe\Documents\GitHub\FaceNet\modelo\svm_model_160x160.pkl'
with open(model_path,'wb') as f:
    pickle.dump(model,f)
