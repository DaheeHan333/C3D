from keras.optimizers import SGD,Adam
from keras.utils import np_utils
import tensorflow as tf
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation
from keras.regularizers import l2
from keras.models import Model
import numpy as np
import numpy as np
import random
import cv2
import os
import random
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #CPU


def process_batch(num, img_path, train=True):
    batch = np.zeros((1,16,112,112,3))#batch, frame
    imgs = os.listdir(img_path)
    imgs.sort(key=str.lower)

    if train:
        crop_x = random.randint(0, 15)
        crop_y = random.randint(0, 58)
        is_flip = random.randint(0, 1)
        for j in range(16):
            img = imgs[j+num*16]
            image = cv2.imread(img_path + '/' + img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (171, 128))
            if is_flip == 1:
                image = cv2.flip(image, 1)
            batch[0][j][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
    else:
        for j in range(16):
            img = imgs[j]
            image = cv2.imread(img_path + '/' + img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (171, 128))
            batch[0][j][:][:][:] = image[8:120, 30:142, :]

    return batch

def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs



def main():
    img_path = '/home/proj_vode/jpg_dataset/kidnap/normal'


    action_list = os.listdir(img_path)
    num_iter = len(action_list)

    print(num_iter)

    weight_decay = 0.005
    num_classes = 2
    batch_size = 32
    #epochs = 16

    layer1 = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))
    layer2 = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')
    layer3 = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))
    layer4 = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')
    layer5 = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))
    layer6 = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')
    layer7 = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))
    layer8 = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')
    layer9 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))
    layer10 = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')
    layer11 = Flatten()
    layer12 = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))
    #layer13 = Dropout(0.5)
    layer14 = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))
    

    
    for i in range(num_iter):
        new_line = []
        for j in range(32):
            y_test = process_batch(j,img_path+'/'+action_list[i],train=True)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            y = layer14(layer12(layer11(layer10(layer9(layer8(layer7(layer6(layer5(layer4(layer3(layer2(layer1(x)))))))))))))
            y = y.numpy()
            new_line.extend(y)
        print(new_line)
        print(len(new_line))
        np.save('/home/proj_vode/numpy/normal/'+action_list[i],new_line)

if __name__ == '__main__':
    main()
