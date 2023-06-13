import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

st.markdown("# MLP")
# Load the dataset
dataset_path = "https://raw.githubusercontent.com/AndrianPrakasa/UAS/main/CSV/csa.csv"
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
history = ann.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0)
ann.fit(x_train, y_train, batch_size=32, epochs=100)
loss, accuracy = ann.evaluate(x_test, y_test)

st.write(f"Test Loss: {loss:.4f}")
st.write(f"Test Accuracy: {accuracy:.4f}")

st.markdown("# Naive Bayess")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Load the dataset
dataset_path = "https://raw.githubusercontent.com/AndrianPrakasa/UAS/main/CSV/csa.csv"
data = pd.read_csv(dataset_path)
data.replace("Agree", 1, inplace=True)
data.replace("Yes", 1, inplace=True)
data.replace("No", 0, inplace=True)
data.replace("Disagree", 0, inplace=True)

# Display the dataset in Streamlit
st.dataframe(data.head())

X = data.iloc[:, :5].values
y = data['Knowledge Level'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy score
st.write(f"Accuracy: {accuracy}")

df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
st.dataframe(df)

st.markdown("# Tree")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the data
dataset = "https://raw.githubusercontent.com/AndrianPrakasa/UAS/main/CSV/csa.csv"
df = pd.read_csv(dataset)
df.replace(["Agree", "Yes"], 1, inplace=True)
df.replace(["No", "Disagree"], 0, inplace=True)

# Split the data into features (X) and target variable (y)
X = df.iloc[:, :5].values
y = df['Knowledge Level'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the decision tree classifier with maximum depth of 3 and Gini impurity as the splitting criterion
clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0)

# Fit the model
clf_gini.fit(X_train, y_train)

# Calculate accuracy on the training set
y_pred_train_gini = clf_gini.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train_gini)

# Calculate accuracy on the testing set
y_pred_gini = clf_gini.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_gini)

# Display the accuracy scores
st.write("Accuracy on training set (Gini): {:.3f}".format(train_accuracy))
st.write("Accuracy on testing set (Gini): {:.3f}".format(test_accuracy))

# Display the decision tree visualization
fig = plt.figure(figsize=(12, 8))
tree.plot_tree(clf_gini.fit(X_train, y_train))
st.pyplot(fig)




