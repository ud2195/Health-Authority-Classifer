# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:00:07 2019

@author: udayk
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:09:31 2019

@author: udayk
"""

import os
import numpy as np
import pandas as pd
import scipy
import sklearn
import keras
from keras.models import Sequential
import cv2
from skimage import io
from matplotlib import pyplot as plt



file=pd.read_excel(r'D:\newf.xlsx')
file=file.loc[file['Communication Format']=='Letter (hardcopy)']

aut=list(set(file['Health Authority']))
import imageio

DIR='D:\\all health authorities\\'
images=[]
label=[]
for x in aut:
    nd=DIR+x+'\\'
    print(nd)
    try:
        for y in os.listdir(nd):
            print(y)
            image=nd+y
            images.append(image)
            label.append(x)
    except Exception as e:
        print(e)
        pass


df=pd.DataFrame({'Path':images,'Category':label}) 
df['Category'].value_counts().sort_values()

df=df.loc[df['Category'].isin(['Minsterio de Salud','Egyptian Drug Authority','Administracion Nacional de Medicamentos, Alimentos y Tecnologia Medica (ANMAT)','Agence nationale de sécurité du médicament et des produits de santé'])]



from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import  preprocess_input
from keras.applications.vgg16 import VGG16

CLASSES = 4
    

base_model = VGG16(weights='imagenet', include_top=False)
from sklearn.preprocessing import OneHotEncoder
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])        

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Category']= encoder.fit_transform(df['Category'])



from keras.preprocessing.image import ImageDataGenerator

WIDTH = 256
HEIGHT = 256
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)



validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


df['Category'] =df['Category'].astype(str)


from sklearn.utils import shuffle
df = shuffle(df)

from sklearn.model_selection import train_test_split

traindf,testdf=train_test_split(df,test_size = 0.2, random_state = 0)

valdf,acttestdf=train_test_split(testdf,test_size = 0.1, random_state = 0)

train_generator = train_datagen.flow_from_dataframe(traindf,target_size=(HEIGHT, WIDTH),batch_size=BATCH_SIZE,class_mode='categorical', x_col='Path', y_col='Category')
    
validation_generator = validation_datagen.flow_from_dataframe(valdf,target_size=(HEIGHT, WIDTH),batch_size=BATCH_SIZE,class_mode='categorical', x_col='Path', y_col='Category')



EPOCHS = 5
BATCH_SIZE = 32
STEPS_PER_EPOCH = 26
VALIDATION_STEPS = 6

MODEL_FILE = 'filename.model'

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)

history.history 

model.save(r'D:\cnnhealthauthority.model')

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()


