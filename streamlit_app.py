import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

# Load the dataset
dataset_path = "path/to/dataset.csv"
data = pd.read_csv(dataset_path)
data.replace("Agree", 1, inplace=True)
data.replace("Yes", 1, inplace=True)
data.replace("No", 0, inplace=True)
data.replace("Disagree", 0, inplace=True)
data.replace("Beginner", 0, inplace=True)
data.replace("Intermediate", 1, inplace=True)

# Display the dataset in Streamlit
st.dataframe(data.head())

x = data.iloc[:, :5].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.45, random_state=1)

# Build the model
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile and train the model
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=32, epochs=100)
