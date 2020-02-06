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


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()




















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
    
# setup model
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
#dfval['Category'] = dfval['Category'].astype(str)


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

model.save(r'D:\90.4filename.model')

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()


from keras.preprocessing import image
model11 = load_model(r'D:\filename.model')
from keras.models import load_model


m=load_model(r'D:\90.4filename.model')


v=[]
def predict(model, img):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
    Returns:
        list of predicted labels and their probabilities 
    """

    y = image.img_to_array(x)
    y = np.expand_dims(y, axis=0)
    y = preprocess_input(y)
    preds = m.predict(y)
    preds=max(preds)
    labels = np.argmax(preds, axis=-1)
    labels=encoder.inverse_transform([int(labels)])
    v.append(labels)
    return labels


for x in acttestdf['Path']:
    predict(model11,x)

v=[]
for x in acttestdf['Path']:
    img = image.load_img(x, target_size=(HEIGHT, WIDTH))
    y = np.expand_dims(img, axis=0)
    y = preprocess_input(y)
    preds = model11.predict(y)
    preds=max(preds)
    labels = np.argmax(preds, axis=-1)
    v.append(labels)


b=list(acttestdf['Category'])
s=[]
for z in b:
    c=int(z)
    s.append(c)


from sklearn.metrics import accuracy_score
print('accuracy %s' % accuracy_score(b, v))

from sklearn.metrics import classification_report
print(classification_report(b, v))
from sklearn.externals import joblib

joblib.dump(encoder,r'D:\30-09-2019-encoderfile')

enn=joblib.load(r'D:\30-09-2019-encoderfile')
acttestdf['Category']=enn.transform(acttestdf['Category'])


from nltk.corpus import wordnet
syns = wordnet.synsets("rome.n.01")
print(syns.lemma_names())
import spacy
nlp = spacy.load('en', parser=False)
def get_related(word):
  filtered_words = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
  similarity = sorted(filtered_words, key=lambda w: word.similarity(w), reverse=True)
  return similarity[:10]
print([w.lower_ for w in get_related(nlp.vocab['jet'])])


from nltk.corpus import wordnet as wn
vehicle = wn.synset('lively.v.02')
typesOfVehicles = list(set([w for s in vehicle.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))

for ss in wn.synsets('meat'):
    print(ss.name(), ss.lemma_names())
    
    
    
from PyDictionary import PyDictionary

dictionary=PyDictionary()    
print (dictionary.synonym("lively"))


import requests
from bs4 import BeautifulSoup

def synonyms(term):
    response = requests.get('http://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.text, 'html')
    section = soup.find('section', {'class': 'synonyms-container'})
    return [span.text for span in section.findAll('span')]



synonyms('voyage')

import RAKE
The lack of career -- based courses in US high schools

from rake_nltk import Rake
r = Rake(max_length=1)

r.extract_keywords_from_text('encourage their students to seek advice about depression')
c
c=r.get_ranked_phrases()
c

for x in 
