
import matplotlib.pyplot as plt
from plot_keras_history import show_history, plot_history
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import itertools
from tensorflow.keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
#import kerastuner as kt
import keras_tuner as kt
from save_plt import save_plot

"""

"""

skin_df = pd.read_csv('../ML/dataverse_files/HAM10000_metadata.csv')


lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
}


# Creating New Columns for better readability
img_directory = '../ML/dataverse_files/HAM10000_images'
img_path = {os.path.splitext(os.path.basename(img))[0]: img for img in glob(os.path.join(img_directory, '*.jpg'))}
skin_df['path'] = skin_df['image_id'].map(img_path.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes


# uncomment this to see the missing values 
#print(skin_df.isnull().sum())

# filling null values for age with the mean
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)

# uncomment to see the result of filling 
#print(skin_df.isnull().sum())
#print(skin_df.dtypes)


fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
#plt.show()

# resize 
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))


# Checking the image size distribution
skin_df['image'].map(lambda x: x.shape).value_counts()

features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)


x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std

# ne-hot encoding on the labels
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)

# Reshape to 3d
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))


input_shape = (75, 100, 3)
num_classes = 7
epochs = 25
batch_size = 10

print("data augmentation")
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

def modified_alexnet(hp):
    AlexNet = Sequential()
    AlexNet.add(Conv2D(
        hp.Int("conv_1", min_value=96, max_value=256, step=32),
        kernel_size=(11, 11), strides=(4, 4), padding="same", input_shape=(75, 100, 3)))
    AlexNet.add(Activation('relu'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    
    AlexNet.add(Conv2D(
		hp.Int("conv_2", min_value=256, max_value=512, step=32),
		kernel_size=(5, 5), strides=(1,1), padding="same"))
    AlexNet.add(Activation('relu'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    
    AlexNet.add(Conv2D(
		hp.Int("conv_3", min_value=384, max_value=512, step=32),
		kernel_size=(3,3), strides=(1,1), padding="same"))
    AlexNet.add(Activation('relu'))
    AlexNet.add(BatchNormalization())

    AlexNet.add(Conv2D(
		hp.Int("conv_4", min_value=384, max_value=512, step=32),
		kernel_size=(3,3), strides=(1,1), padding="same"))
    AlexNet.add(Activation('relu'))
    AlexNet.add(BatchNormalization())

    AlexNet.add(Conv2D(
		hp.Int("conv_5", min_value=256, max_value=512, step=32),
		kernel_size=(3, 3), strides=(1,1), padding="same"))
    AlexNet.add(Activation('relu'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))


    AlexNet.add(Flatten())
    
    AlexNet.add(Dense(hp.Int(name='units_1',min_value=4096, max_value=8192, default=4096,step=256) , input_shape=(32,32,3,)))
    AlexNet.add(Activation('relu'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Dropout(0.4))
    AlexNet.add(Dropout(hp.Float('dropout_1', 0, 0.4, step=0.1, default=0.4)))


    AlexNet.add(Dense(hp.Int(name='units_2',min_value=4096, max_value=8192, default=4096,step=256 )))
    AlexNet.add(Activation('relu'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Dropout(0.4))
    AlexNet.add(Dropout(hp.Float('dropout_2', 0, 0.4, step=0.1, default=0.4)))


    # todo: remove this ??
    AlexNet.add(Dense(1000))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(Dropout(0.4))


    
    # Output 
    AlexNet.add(Dense(7))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('softmax'))  
  
  
    # initialize the learning rate choices
    learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
    #learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling="log")

    # compile the model
    AlexNet.compile(optimizer = Adam(learning_rate = learning_rate), loss = "categorical_crossentropy", metrics=["accuracy"])
	
    return AlexNet
    

print("instantiating a bayesian optimization tuner object")
tuner = kt.BayesianOptimization(
        modified_alexnet,
        objective="val_accuracy",
		max_trials=10,
		seed=42,
        directory='bayesian_opt',
        project_name='alx_opt')
        
# early stop callback to avoid overfitting
#test different patience vals??
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tuner.search_space_summary()

tuner.search(
	x=x_train, y=y_train,
	validation_data=(x_validate, y_validate),
	#batch_size=batch_size,
	#callbacks=[es],
	epochs=epochs
)


bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
tuner.results_summary()

print("training the best model")
AlexNetOpt = tuner.hypermodel.build(bestHP)

hostory_opt = AlexNetOpt.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])



modified_alexnet_val_acc = max(hostory_opt.history['val_accuracy'])

loss, accuracy = AlexNetOpt.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = AlexNetOpt.evaluate(x_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))


# instanciate AlexNet
AlexNet = Sequential()

#1st CNN Layer
AlexNet.add(Conv2D(filters=96, input_shape=(75,100,3), kernel_size=(11,11), strides=(4,4), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#2nd CNN layer
AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#3rd CNN layer
AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#4th CNN layer
AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))

#5th CNN layer
AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

# followed by 3 fully Connected layers
AlexNet.add(Flatten())
# 1st fully connected layer
AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
# Add Dropout to prevent overfitting
#AlexNet.add(Dropout(0.3)) 
AlexNet.add(Dropout(0.4))

#2nd fully connected layer
AlexNet.add(Dense(4096))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(Dropout(0.4))

#3rd fully connected layer
AlexNet.add(Dense(1000))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('relu'))
AlexNet.add(Dropout(0.4))

# Output 
AlexNet.add(Dense(7))
AlexNet.add(BatchNormalization())
AlexNet.add(Activation('softmax'))

# get summary for the model
AlexNet.summary()

# optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile the model
AlexNet.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# Fit the model
history = AlexNet.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])



loss, accuracy = AlexNet.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = AlexNet.evaluate(x_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))



# storing baseline accuracy (no optimization)
baseline_acc = max(history.history['val_accuracy'])

print("The validation accuracy of the baseline alexnet model is {} VS The validation accuracy of the modified baseline alexnet model is {}".format(baseline_acc, modified_alexnet_val_acc))

save_plot(history, 'alx_base')
save_plot(hostory_opt, 'alx_opt')






