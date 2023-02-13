# Convolutional Neural Networks (CNN)

import tensorflow as tf

# Keras https://keras.io API built on top of Tensor Flow 2

from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# ------------------------------- Part 1 - Data pre-processing -------------------------------

# processing the training set
# apply transformation on training set to over feeding, augument diversity of the images
# Initiated an object of class Image Data Generator

# https://keras.io/api/preprocessing/image/

train_datagen = ImageDataGenerator(
    rescale = 1/.255,
    shear_range= 0.2,
    zoom_range= 0.2,
    horizontal_flip= True)

# call method from the object
train_set = train_datagen.flow_from_directory(
    'data/training_set',
    target_size= (64, 64),
    batch_size = 32,
    class_mode = 'binary')

# processing the test set

test_datagen = ImageDataGenerator(rescale=1/.255)

test_set = test_datagen.flow_from_directory(
    'data/test_set',
    target_size= (64, 64),
    batch_size= 32,
    class_mode= 'binary'
)

# ------------------------------- Part 2 - Building CNN -------------------------------

# initialize the CNN model
cnn = tf.keras.models.Sequential()

# Kernel size is the size of filter detector

cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size= 3,
    activation= 'relu',
    input_shape= [64, 64, 3])
)


# ------------------------------- Part 3 - Training CNN -------------------------------


# ------------------------------- Part 4 - Preditcion -------------------------------
