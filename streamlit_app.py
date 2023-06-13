# -*- coding: utf-8 -*-
"""MLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SzdHS6rw9lykY9EhXf0A9N3bRphiq5Ft
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

from google.colab import drive
drive.mount('/content/drive')

dataset = "/content/drive/MyDrive/pendat/csa.csv"
data = pd.read_csv(dataset)
data.replace("Agree", 1 ,inplace=True)
data.replace("Yes", 1 ,inplace=True)
data.replace("No", 0 ,inplace=True)
data.replace("Disagree", 0 ,inplace=True)
data.replace("Beginner", 0 ,inplace=True)
data.replace("Intermediate", 1 ,inplace=True)
st.data.head()

x = data.iloc[:,:5].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.45, random_state = 1)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

ann.fit(x_train,y_train,batch_size=32,epochs=100)
