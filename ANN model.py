#Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import os
from glob import glob
import cv2
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras import backend as K
from tensorflow.keras.optimizers import Adam

#read data
df = pd.read_csv('..\ML\dataverse_files\HAM10000_metadata')

#better column names
type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
df['diagnosis'] = df['dx'].map(type_dict.get) 
df['diagnosis_id'] = pd.Categorical(df['diagnosis']).codes

#data cleaning

# replacing null values with the mean age
df['age'].fillna((df['age'].mean()), inplace=True)
df.isnull().sum()

df = df[df['sex'] != 'unknown']

#deleting duplicates?

#data balancing

#mapping the right images to the right Lesion ID's
img_directory = '..\ML\dataverse_files\HAM10000_images'
img_path = {os.path.splitext(os.path.basename(img))[0]:img for img in glob(os.path.join(img_directory, '*.jpg'))}
df['path'] = df['image_id'].map(img_path.get)
df['image'] = df['path'].map(lambda z: np.asarray(Image.open(z).resize((100,100))))

features = df.drop(columns=['diagnosis_id'], axis=1)
diagnosis_target = df['diagnosis_id']
x_train1, x_test1, y_train1, y_test1 = train_test_split(features, diagnosis_target, test_size=0.2)
features.head()

tf.unique(x_train1.diagnosis.values)


x_train = np.asarray(x_train1['image'].tolist())
x_test = np.asarray(x_test1['image'].tolist())

# normalizing data?
x_train_mean = np.mean(x_train)
x_test_mean = np.mean(x_test)
x_train_std = np.std(x_train)
x_test_std = np.std(x_test)
x_train = (x_train - x_train_mean)/ x_train_std
x_test = (x_test - x_test_mean)/ x_test_std

# Perform one-hot encoding on the labels
y_train = to_categorical(y_train1, num_classes = 7)
y_test = to_categorical(y_test1, num_classes = 7)

# splitting training and validation
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1)

# re-shaping the images in 3 dimensions
x_train = x_train.reshape(x_train.shape[0], *(100, 100, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(100, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(100, 100, 3))

x_train = x_train.reshape(7169,100*100*3)
x_test = x_test.reshape(1992,100*100*3)
print(x_train.shape)
print(x_test.shape)

# define the keras model
model = Sequential()

model.add(Dense(units= 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30000))
model.add(Dense(units= 32, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units= 32, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units= 32, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))

optimizer = Adam(learning_rate = 0.001,beta_1 = 0.9,beta_2 = 0.999,epsilon = 1e-8)

#  compile the keras model
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# fit the keras model on the dataset
history = model.fit(x_train, y_train, batch_size = 10, epochs = 50)

accuracy = model.evaluate(x_test, y_test, verbose=1)[1]
print("Test: accuracy = ",accuracy*100,"%")

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
