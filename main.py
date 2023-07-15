import numpy as np
import os
import csv
import pandas as pd
import tensorflow

folders = ['./pictures/flower_photos/daisy',
           './pictures/flower_photos/dandelion',
           './pictures/flower_photos/roses',
           './pictures/flower_photos/sunflowers',
           './pictures/flower_photos/tulips']
folder_test = './pictures/test-images'
output_file = './flowers.csv'
output_file_test = './flowers_test.csv'
numar =0
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




if __name__ == '__main__':
   # createcsv_for_train()
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
   print(test_labels)







