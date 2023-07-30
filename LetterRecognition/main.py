import os
import random
import tensorflow as tf
import numpy as np


folders = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
root = "."
trainFolder = "/lettersTrain"
testFolder = "/lettersTest"


def getListOfImages(fromFolder):
    global folders
    global root

    allImagesArray = np.array([], dtype=str)
    allImagesLabelsArray = np.array([], dtype=str)

    for folder in folders:
        print("Loading Image Name of ", folder)
        currentAlphabetFolder = root + fromFolder + "/" + folder
        imagesName = os.listdir(currentAlphabetFolder)
        allImagesArray = np.append(allImagesArray, imagesName)  # append all names of images (feature)
        print("Nr. of images: ", len(imagesName))
        for i in range(0, len(imagesName)):
            allImagesLabelsArray = np.append(allImagesLabelsArray,
                                             currentAlphabetFolder)  # append name of folder to each image (labels)
    return allImagesArray, allImagesLabelsArray


def shuffleImagesPath(imagesArray, labelsArray):
    print("Size to shuffle: ", len(imagesArray))
    for i in range(0, 100000):

        randomIndex1 = random.randint(0, len(imagesArray)-1)
        randomIndex2 = random.randint(0, len(imagesArray)-1)

        imagesArray[randomIndex1], imagesArray[randomIndex2] = imagesArray[randomIndex2], imagesArray[randomIndex1]
        labelsArray[randomIndex1], labelsArray[randomIndex2] = labelsArray[randomIndex2], labelsArray[randomIndex1]
    print("Shuffling done")
    return imagesArray, labelsArray

#Create a neuronal network: Network Parameters (784-100-100-100-10)
n_input = 784
hidden_layer_neurons = 100
hidden_layer2_neurons = 100
hidden_layer3_neurons = 100
n_classes = 10
tf.compat.v1.disable_eager_execution()
# ********** crearea retelei-> graful de invatare
x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_input])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, n_classes])


print("Weight-uri")
w1 = tf.Variable(tf.random.normal([n_input, hidden_layer_neurons]))
w2 = tf.Variable(tf.random.normal([hidden_layer_neurons, hidden_layer2_neurons]))
w3 = tf.Variable(tf.random.normal([hidden_layer2_neurons, hidden_layer3_neurons]))
w4 = tf.Variable(tf.random.normal([hidden_layer3_neurons, n_classes]))



b1 = tf.Variable(tf.random.normal([hidden_layer_neurons]))
# array de 100  linii
b2 = tf.Variable(tf.random.normal([hidden_layer2_neurons]))
# array de 100  linii
b3 = tf.Variable(tf.random.normal([hidden_layer3_neurons]))
# array de 100  linii
b4 = tf.Variable(tf.random.normal([n_classes]))
# array de 10  linii


# Construirea si aplicraea functiilor de activare pt fiecare layer: f = w*x +b

hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer, w2), b2))
hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer2, w3), b3))
output_layer = tf.add(tf.matmul(hidden_layer3, w4), b4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
print(f'Cost: {cost}')
regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)+tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
print(f'regularizers: {regularizers}')
cost = tf.reduce_mean(cost + 1*regularizers)
print(f'Cost: {cost}')

# training parameters
learning_rate = 0.02
epochs = 5
batch_size = 300
nrBatches = 300*10//batch_size

trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Define the Test model and accuracy
#argmax: vector cu pozitiilr pe care s-a obtinut probabitatea cea mai mare(clasofocarea buna
#tf.equal: va genera un vecor de 0 si 1. daca in realitate e ok si prezis corecct =1, altfel 0
correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))

# reduce_mean = aplica functia de acuratete:
# ex: [1  0 1 1]=> mean = (1+0+1+1)/4
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#*********** preluarea caracteristicilor setului de date- pentru TEST ********************


#Preluarea imaginilor - nu am folosit batch-uri!
# rationalentul e ca la test
def getBatchOfLetterImages():
    global imagesArray
    global labelsArray

    features = np.ndarray(shape=(0, 784), dtype=np.float32)
    labels = np.ndarray(shape=(0, 10), dtype=np.float32)

    with  tf.compat.v1.Session() as sess:
        for i in range(0, len(imagesArray)):
            # pas1: path-ul pentru imagine in folder
            pathToImage = labelsArray[i]+imagesArray[i]
            lastIndexOfSlash = pathToImage.rfind("/")
            folder = pathToImage[lastIndexOfSlash - 1]
            # pas2: se iau info din imagine
            imageContents = tf.io.read_file(str(pathToImage))
            image = tf.image.decode_png(imageContents, dtype=tf.uint8)
            #pas3: redimensionare imagine
            resized_image = tf.image.resize(image, [28, 28])
            imarray = resized_image.eval()
            imarray = imarray.reshape(784)
            #pas4: adaug noile info:
            # -->label: array cu 1 pe pozitia literei
            # -->features: array de 784 pozitii,  cu datele redimensionate

            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([getOneHot(folder)], dtype=np.float32)
            labels = np.append(labels, appendingNumberLabel, axis=0)
            features = np.append(features, appendingImageArray, axis=0)

    return labels, features

def getOneHot(alphabet):
    matrice = np.eye(10, dtype=np.float32)
    return np.eye(10, dtype=np.float32)[ord(alphabet) - ord('A')]

def getTestArrays(testImages, testLabels):
    # se vor genera vectori in care sa se tina pixelii unei imagini introduse: 784 elemnetr
    dataset = np.ndarray(shape=(0, 784), dtype=np.float32)
    # etichete vor fi 10, pentru fiecare litera -> fiind o combinatie cu 1 pe pozitia literei, in rest, 9 pozitii cu 0
    labels = np.ndarray(shape=(0, 10), dtype=np.float32)
    with tf.compat.v1.Session() as sess:
        for i in range(0, len(testImages)):
            pathToImage = testLabels[i]+'/'+testImages[i]
            #print(pathToImage) : ./lettersTest/ERnJlZWRvbSBFeHRlbmRlZCBJdGFsaWMudHRm.png
            lastIndexOfSlash = pathToImage.rfind("/")
            folder = pathToImage[lastIndexOfSlash - 1]


            #prelucare imagine:
            #pas1: citesc imagine de la path:

            imageContents = tf.io.read_file(str(pathToImage))

            #pas2: decodificarea in png:
            image = tf.image.decode_png(imageContents, dtype=tf.uint8)

            #pas3: redimensionam imaginea: (in cazul in care sunt de dimensiuni diferite,sa devina toate identice)
            resized_image = tf.image.resize(image, [28, 28])

            imarray = resized_image.eval()
            imarray = imarray.reshape(784)

            appendingImageArray = np.array([imarray], dtype=np.float32)
            appendingNumberLabel = np.array([getOneHot(folder)], dtype=np.float32)
            labels = np.append(labels, appendingNumberLabel, axis=0)
            dataset = np.append(dataset, appendingImageArray, axis=0)



    print(f'Primul el din setul de date: {dataset}')
    print(f'Prima eticheta din setul de date: {labels}')
    return dataset, labels


#after each batch the value remains because it is a global variable
batchIndex = 0
imagesArray, labelsArray = getListOfImages(trainFolder)
imagesArray, labelsArray = shuffleImagesPath(imagesArray, labelsArray)
testImages, testLabels = getListOfImages(testFolder)
shufTestData, shufTestLabel = shuffleImagesPath(testImages,testLabels)
testData, testDataLabel  = getTestArrays(shufTestData, shufTestLabel)

if __name__ == "__main__":
    allImagesArray, allImagesLabelsArray= getListOfImages("/lettersTrain")
    #printare random:
    r1 = random.randint(0, allImagesArray.size)
    r2 = random.randint(0, allImagesArray.size)
    print(f'Before, {r1}, {r2}')
    print(f'nume imagine {allImagesArray[r1]}') #QmFja3RhbGtTZXJpZiBCVE4gU0MgQm9sZE9ibGlxdWUudHRm.png
    print(f'path imagine {allImagesLabelsArray[r2]}') #./lettersTrain/H
    shuffleImagesPath(allImagesArray, allImagesLabelsArray)
    print(f'After, {r1}, {r2}')
    print(f'nume imagine {allImagesArray[r1]}') #QmFja3RhbGtTZXJpZiBCVE4gU0MgQm9sZE9ibGlxdWUudHRm.png
    print(f'path imagine {allImagesLabelsArray[r2]}') #./lettersTrain/H
    print(f'Bias\nb1: {b1}\nb2: {b2}\nb3: {b3}\nb4: {b4}')
    print(f'Weight-uri\nw1: {w1}\nw2: {w2}\nw3: {w3}\nw4" {w4}')
    print("Hidden Layer")
    print(tf.matmul(x, w1))
    print( tf.add(tf.matmul(x, w1), b1))
    print(hidden_layer)
    print("Hidden Layer 2")
    print(hidden_layer2)
    print("Hidden Layer 3")
    print(hidden_layer3)
    print("Output layer")
    print(output_layer)
    print(f'Acurtetea: {accuracy}')

    # ANtrenare si test
    with  tf.compat.v1.Session() as session:
        for i in range(0, epochs):
            for j in range(0, 5):
                batchX = getBatchOfLetterImages()
                opt = session.run(trainer, feed_dict={x: batchX})
                loss, acc = session.run([cost, accuracy], feed_dict={x: batchX})
                print("Iteration: " + str(j + 1) + ", Loss= " + "{:.6f}".format(
                    loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            print(
                "Epoch: " + str(i + 1) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(
                    acc))
        print("Test accuracy: ", accuracy.eval(feed_dict={x: testData, y: testDataLabel}))


