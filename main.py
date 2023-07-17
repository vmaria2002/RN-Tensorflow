# surse:
# [1] https://github.com/ArdeleanRichard/University-Projects/tree/main/Python/MachineLearning/FlowerRecognition
# [2] https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/image_feature_vector.ipynb
# [3] https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/5
# https://www.tensorflow.org/

import numpy as np
import os
import csv
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import math
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.estimator import keras

folders = ['./pictures/flower_photos/daisy',
           './pictures/flower_photos/dandelion',
           './pictures/flower_photos/roses',
           './pictures/flower_photos/sunflowers',
           './pictures/flower_photos/tulips']
folder_test = './pictures/test-images'
output_file = './flowers.csv'
output_file_test = './flowers_test.csv'
numar =0

# Date adaptate pt. imbunatirea invatarii
global learning_rate
global epochs
global batch_size
global train_images
global test_images
global features
global labels
#[1]
global n_input
global hidden_layer_neurons
global hidden_layer2_neurons
global hidden_layer3_neurons
global n_classes

#[1]
n_input = 2352
hidden_layer_neurons = 100
hidden_layer2_neurons = 50
hidden_layer3_neurons = 25
n_classes = 5

learning_rate= 0.1
epochs= 5
counter =0
# batch_size=[0,0,0,0,0](sau mai frumos):
batch_size =np.zeros(5)
def createcsv_for_train():
    #deschidere fisier .csv
    new_file = open(output_file, 'w')
    # antet: Scrie antetul în fișierul CSV
    with open(output_file, 'w') as csvfile:
        writer = csv.writer(csvfile)

        # Scrie antetul în fișierul CSV
        writer.writerow(['file', 'label'])

        for subfolder in folders:
            photo = os.listdir(subfolder)
            # scriere date:
            for i in photo:
                writer.writerow([i, folders.index(subfolder)])
                #print(folders.index(subfolder))
                batch_size[folders.index(subfolder)]=batch_size[folders.index(subfolder)]+1

def createcsv_for_test():
    new_file = open(output_file_test, 'w')
    with open(output_file_test, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file', 'label'])

        photo = os.listdir(folder_test)
        # scriere date:
        for i in photo:
            writer.writerow([i])

def prelucrare_date_test():
    print("\n***************** Test **********************\n")

    data_test= pd.read_csv(output_file_test, header=1, names=['file','label'])

    print("Numar de inregistrari in .csv:")
    print(data_test.shape)

    print("Primele 5 inregistrari")
    print(data_test.head())

    null_elements = data_test.isnull().sum().sum()
    print(f'In test dataset exists,{null_elements} null elements.')
    return  data_test

def preluare_date_train():
    print("************ Train ************************")

    data_train = pd.read_csv(output_file,header=1, names=['file','label'])
    print("numarul de linii si coloane")
    print(data_train.shape)

    print("afisarea primelor 5 linii de date:")
    print(data_train.head())

    #numar = input("accesare record-ului numarul: ")
    #print(data_train.iloc[int(numar)])

    print("informatii despre datele din csv:")
    print(data_train.info())


    print("Verific daca exista valori nule in fiecare coloana:")
    valori_nule = data_train.isnull().sum().sum()
    if valori_nule == 0:
        print("Doesn't exit null value in data set")
    else:
       print(f'In data set exists {valori_nule} null values')

    return data_train




def matrix():
    print('matrice patratica')

    #crearea unei matrici 5x5(unitate/identitate)
    mat = np.eye(5, dtype=np.float32)
    print(mat)

    #pentru accesarea unei linii di matrice:
    line_2 = np.eye(5, dtype=np.float32)[1]
    print(line_2)

    return mat, line_2



#[2]
def display_images(images, labels, num_columns=5):
    num_rows = math.ceil(len(images) / num_columns)
    fig = plt.figure(figsize=(num_columns * 3, num_rows * 3))

    for i, image_path in enumerate(images):
        ax = fig.add_subplot(num_rows, num_columns, i + 1)
        ax.axis('off')

        image = Image.open(image_path)
        ax.imshow(image)
        ax.set_title(labels[i])

    plt.tight_layout()
    plt.show()


def select_image():

    # Example usage
    nr = input("introduceti tipul de floare\n0 - pentru daisy\n1 - pentru dandelion\n2 - pentru roses\n3 - pentru sunflowers\n4- tulips ")
    i_nr = int( nr )

    b=0
    if i_nr ==0:
        max_poz =0
        max= batch_size[0]
    else:
        max_poz= sum(batch_size[:i_nr-1])
        max =sum(batch_size[:i_nr])

    poz= input(f'inserati o valoare intre {max_poz}-{max}')

    print(folders[i_nr])
    image_paths = folders[i_nr]+'/'+train_images[int(poz)]
    print(image_paths)

    display_images([image_paths], [image_paths])

#[2][1] build model:
startIndexOfBatch = 0

# aici trebuie sa corectezi, sa ia path-ul corect!!
# iar la labels tb sa vezi cum a facut Richard - ca ia pt fiecare label - construtia"
# ex: 0:[1,0,0,0,0], 1:[0, 1,0,0,0]....

def decode_and_resize_image(encoded):
    features = np.ndarray(shape=(0, 3669), dtype=np.float32)
    labels = np.ndarray(shape=(0, 5), dtype=np.float32)

    with tf.Session() as sess:
        for l in len(folders):
            for i in range(startIndexOfBatch, len(train_images)):
                pathToImage = folders[l] + train_images[i]
                imageContents = tf.read_file(str(pathToImage))

                image = tf.image.decode_jpeg(imageContents, channels=3)
                decoded = tf.image.convert_image_dtype(image, tf.float32)
                resized_image = tf.image.resize_images(decoded, [28, 28])

                return resized_image

global encoded_images
global image_size
global batch_images
global features
global image_module



# [2]
def decode_and_resize_image(encoded):
  decoded = tf.image.decode_jpeg(encoded, channels=3)
  decoded = tf.image.convert_image_dtype(decoded, tf.float32)
  return tf.image.resize_images(decoded, image_size)




# Load a pre-trained TF-Hub module for extracting features from images. We've
# chosen this particular module for speed, but many other choices are available.


image_module = "https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2"
image_module = hub.load(image_module)
# Preprocessing images into tensors with size expected by the image module
# Preprocessing images into tensors with size expected by the image module.

encoded_images =tf.keras.Input(shape=[None], dtype=tf.string)
image_size = [27,27]

# **************** Betch images - R,A ******************

def decode_and_resize_image(encoded):
  decoded = tf.image.decode_jpeg(encoded, channels=3)
  decoded = tf.image.convert_image_dtype(decoded, tf.float32)
  return tf.image.resize_images(decoded, image_size)

# ************


if __name__ == '__main__':
   createcsv_for_train()
   # createcsv_for_test()
   data_train = preluare_date_train()

   print("Obtinere de array-uri cu coloanele pt. TRAIN data")
   #get_image = data_train['file'][1]
   train_images = data_train['file'].tolist()
   train_labels = data_train['label'].tolist()

   print("Obtinere de array-uri cu coloanele pt. REST data")
   data_test= prelucrare_date_test()
   test_images = data_test['file'].tolist()
   test_labels = data_test['label'].tolist()


   print(f'In setul de train, avem:{batch_size} etichete din fiecare tip de floare')
   print("************** Afisarea unei imagini dorite")
  # select_image()










