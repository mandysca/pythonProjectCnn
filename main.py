# Convolutional Neural Networks (CNN)

import tensorflow as tf

# Keras https://keras.io API built on top of Tensor Flow 2

from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# ------------------------------- Part 1 - Data pre-processing -------------------------------

# processing the training set
# apply transformation on training set to over feeding, augument diversity of the images
# Initiated a object of class Image Data Generator
train_datagen = ImageDataGenerator(
    rescale = 1/.255,
    shear_range= 0.2,
    zoom_range= 0.2,
    horizontal_flip= True)

# call method from the object
train_set = train_datagen.flow_from_directory(
    'data/training_set',
    target_size= (150,150),
    batch_size = 32,
    class_mode = 'binary')

# processing the test set


# ------------------------------- Part 2 - Building CNN -------------------------------


# ------------------------------- Part 3 - Training CNN -------------------------------


# ------------------------------- Part 4 - Preditcion -------------------------------
