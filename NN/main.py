
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#
#
# # implement neural network to recognize handwritten digits
# # Use MNIST data set
#
# MNIST data set is built into keras
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# normalize data
# Question: Why is this important for preprocessing?
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
#
# creating model
model = tf.keras.models.Sequential()

# flatten 28 x 28 pixel images to an array
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))

# adding hidden layers
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))

# output layer, using softmax as activation
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# compile model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# train model
model.fit(x_train, y_train, epochs = 5)
#
# model.save('digit_recog.model')
#
#
#
#
#
#

# model = tf.keras.models.load_model('digit_recog.model')
loss, accuracy = model.evaluate(x_test, y_test)

print(f'The loss function on the test set is: {loss}')
print(f'The accuracy of our predictions is: {accuracy}')