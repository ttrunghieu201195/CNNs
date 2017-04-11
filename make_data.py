import matplotlib.pyplot as plt

import os
import tensorflow as tf
from tqdm import tqdm
from random import shuffle
import cv2 
import numpy as np

folder_dir = '/home/trantrunghieu/lv/project/data/train'
Eval_DIR = '/home/trantrunghieu/lv/project/data/check'
folder_list=[os.path.join(folder_dir,folder) for folder in
                 os.listdir(folder_dir) if not folder.startswith('.')]
test_list = [os.path.join(Eval_DIR,folder) for folder in
                 os.listdir(Eval_DIR) if not folder.startswith('.')]
# check_dir = '/home/trantrunghieu/lv/project/data/test'
check_dir = '/home/trantrunghieu/lv/data/test_random_img'
check_list = [os.path.join(check_dir,folder) for folder in
                 os.listdir(check_dir) if not folder.startswith('.')]
IMG_SIZE = 128

def one_hot(element,list_of_elements):
    ''' ex:- one_hot('C',['A','B','C','D']) returns [0,0,1,0]
       in your case,
       element = absolute path of a subfolder
       list_of_elements = list of folders in main folder i.e os.listdir(main_folder)
   '''
    k=[0 for i in range(len(list_of_elements))]
    index=list_of_elements.index(element)
    k[index]=1
    return k

def create_train_data():
    training_data = []
    for folder in test_list:
    	label = one_hot(folder,test_list)
    	for img in tqdm(os.listdir(folder)):
        		path = os.path.join(folder,img)
        		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        		img = img.reshape(IMG_SIZE*IMG_SIZE)
        		training_data.append([np.array(img),np.array(label)])
	print(folder)
	print(label)
    shuffle(training_data)
    print(training_data[0])
    np.save('eval_data.npy', training_data)
    print("Create train data success!!!"
    return training_data

def process_test_data():
    testing_data = []
    for folder in test_list:
    	for img in tqdm(os.listdir(folder)):
	        path = os.path.join(folder,img)
	        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
	        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
	        img = img.reshape(IMG_SIZE*IMG_SIZE)
	        testing_data.append([np.array(img)])
	print(folder)
    shuffle(testing_data)
    np.save('test_data_random.npy', testing_data)
    print("Create test data success!!!")
    return testing_data

train_data = create_train_data()

test_data = process_test_data()

	
