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
    'dataset/training_set',
    target_size= (64, 64),
    batch_size = 32,
    class_mode = 'binary')

# processing the test set

test_datagen = ImageDataGenerator(rescale=1/.255)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size= (64, 64),
    batch_size= 32,
    class_mode= 'binary'
)

# ------------------------------- Part 2 - Building CNN -------------------------------

# step - 1 : initialize the CNN model
cnn = tf.keras.models.Sequential()

# Kernel size is the size of filter detector

cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size= 3,
    activation= 'relu',
    input_shape= [64, 64, 3])
)

# step - 2 : Pooling
# Max Pooling is used to reduce the size of the feature map
cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2))

# add a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size= 3, activation= 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2))

# step - 3 : Flattening
# convert the pooled feature map into a large feature vector
cnn.add(tf.keras.layers.Flatten()) # input layer

# step - 4 : Full Connection
# add a fully connected layer
cnn.add(tf.keras.layers.Dense(units= 128, activation= 'relu'))

# step - 5 : Output Layer
# add the output layer
cnn.add(tf.keras.layers.Dense(units= 1, activation= 'sigmoid'))

# ------------------------------- Part 3 - Training CNN -------------------------------

# compile the CNN
# optimizer is the algorithm to find the optimal set of weights in the NN that will make the model perform best
# loss function is the loss function within the stochastic gradient descent algorithm
# metrics is the performance metric used to evaluate the model
cnn.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# train the CNN on the training set and evaluate it on the test set
# epochs is the number of times the entire training set passes through the CNN
# x is the training set
# validation_data is the test set
cnn.fit(x= train_set, validation_data= test_set, epochs= 25)

# ------------------------------- Part 4 - Preditcion -------------------------------
# predict if a single image is a dog or a cat
# numpy is a library for scientific computing with Python
import numpy as np

# keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
from keras.preprocessing import image

# load the image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size= (64, 64))

# convert the image to an array
test_image = image.img_to_array(test_image)

# add a dimension to the array
test_image = np.expand_dims(test_image, axis= 0)

# predict the image
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

