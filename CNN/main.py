# This time round, I am going to use a CNN to recognize digits

# import relevant packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# import MNIST data set
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0])

# normalize data; will do this by flattening data and dividing by 255, as RGB values range from 0 to 255



