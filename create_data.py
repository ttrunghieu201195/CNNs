import matplotlib.pyplot as plt

import os
import tensorflow as tf
from tqdm import tqdm
from random import shuffle
import cv2 
import numpy as np



fig=plt.figure()



# ngoc, nguyen, ky, truong, phhuc, duc, huyen, thinh, duyen, hieu, lgiang, giang, hau, my, btran, tram, manh
train_dir = '/home/manhthe/project/dulieu/train'
train_list=[os.path.join(train_dir,folder) for folder in
                 os.listdir(train_dir) if not folder.startswith('.')]

valid_dir = '/home/manhthe/project/dulieu/valid'
valid_list = [os.path.join(valid_dir,folder) for folder in
                 os.listdir(valid_dir) if not folder.startswith('.')]

test_dir = '/home/manhthe/project/dulieu/test'
test_list = [os.path.join(test_dir,folder) for folder in
                 os.listdir(test_dir) if not folder.startswith('.')]

check_dir = '/home/manhthe/Downloads/Desktop/giangoc'
# ngoc_dir = '/home/ngoc/luanvan/huyen'
# ngoc_dir = '/home/ngoc/luanvan/ngoc'
# tram_dir = '/home/ngoc/luanvan/Tram'
# tran_dir = '/home/ngoc/luanvan/tran'
# lgiang_dir = '/home/ngoc/luanvan/Giang'
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
    for folder in train_list:
    	label = one_hot(folder,train_list)
        # print folder
    	for img in tqdm(os.listdir(folder)):
        		path = os.path.join(folder,img)
        		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        		img = img.reshape(IMG_SIZE*IMG_SIZE)
        		training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    # np.save('train_data', training_data)
    print("Create train data success!!!")
    return training_data

def create_valid_data():
    eval_data = []
    for folder in valid_list:
        label = one_hot(folder,valid_list)
        # print folder
        for img in tqdm(os.listdir(folder)):
                path = os.path.join(folder,img)
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                img = img.reshape(IMG_SIZE*IMG_SIZE)
                eval_data.append([np.array(img),np.array(label)])
    shuffle(eval_data)
    # np.save('valid_data', eval_data)
    print("Create valid data success!!!")
    return eval_data



def create_test_data():
    testing_data = []
    for folder in test_list:
        label = one_hot(folder,test_list)
    	for img in tqdm(os.listdir(folder)):
	        path = os.path.join(folder,img)
	        # img_num = img.split('.')[0]
	        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
	        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
	        img = img.reshape(IMG_SIZE*IMG_SIZE)
	        testing_data.append([np.array(img),np.array(label)])
	print(folder)
    shuffle(testing_data)
    # np.save('test_data', testing_data)
    print("Create test data success!!!")
    return testing_data

def test_data_random():
    check_data = []
    for img in os.listdir(check_dir):
        path = os.path.join(check_dir,img)            
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        img = img.reshape(IMG_SIZE*IMG_SIZE)
        check_data.append([np.array(img)])
    print(folder)
    shuffle(check_data)
    # np.save('lgiang_data.npy', check_data)
    print("Create test data success!!!")
    return check_data

# train_data = create_train_data()
# print("Size of train data:",len(train_data))

# eval_data = create_eval_data()
# print("Size of eval data:",len(eval_data))
# # test_data = process_test_data()

# test_data = process_test_data()
# print("Size of test data:",len(test_data))

# random_data = test_data_random()
# print("Size of test data:",len(random_data))
# print(random_data[0])



