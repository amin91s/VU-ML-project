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
from keras.optimizers import adam_v2

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

x_train = np.asarray(x_train1['image'].tolist())
x_test = np.asarray(x_test1['image'].tolist())

# normalizing data?
x_train_mean = np.mean(x_train)
x_test_mean = np.mean(x_test)
x_train_std = np.std(x_train)
x_test_std = np.std(x_test)
x_train = (x_train - x_train_mean)/ x_train_std
x_test = (x_test - x_test_mean)/ x_test_std

# one-hot enconding to the labels
y_train = to_categorical(y_train1, num_classes=7)
y_test = to_categorical(y_test1, num_classes=7)

# splitting training and validation
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1)

# re-shaping the images in 3 dimensions
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))

# convolution neural network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='sigmoid', padding='Same',  input_shape=(75, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='sigmoid', padding='Same',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) # Dropout 25% of the nodes of the previous layer during training

model.add(Conv2D(64, (3, 3), activation='sigmoid', paddin='Same'))
model.add(Conv2D(64, (3, 3), activation='sigmoid', paddin='Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Flatten())     # Flatten, and add a fully connected layer
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.summary()

# Defining the optimizer and compiling the model
optimizer = adam_v2(lr=0.001) # lr is the learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# A learning rate annealer
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=5, min_lr=0.00001)

# adding data augmentation to prevent overfitting
datagenerator = ImageDataGenerator(featurewise_center=False, samplewise=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=20, rescale=1/255, shear_range=0.1,  zoom_range=0.15, width_shift_range=0.15, height_shift_range=0.15, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
datagenerator.fit(x_train)
history = model.fit(datagenerator.flow(x_train, y_train, batch_size=10), epochs=100,  validation_data=(x_validate, y_validate), verbose=1, steps_per_epoch=x_train.shape[0]//10, callbacks=lr_reduction)

# Evaluating the model

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

plt.figure(figsize=(15, 5))
loss = pd.DataFrame(model.history.history)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'r', label='accuracy')
plt.plot(history.history['accuracy_v'], 'b', label='accuracy_v')
plt.set_title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'r', label='loss')
plt.plot(history.history['loss_v'], 'b', label='loss_v')
plt.set_title('Loss accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'])

plt.show()
