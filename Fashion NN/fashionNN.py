import tensorflow as tf
import numpy as np
from tensorflow import keras

# Call Back to stop training when reached a certain % of Accuracy
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if(logs.get('accuracy')>0.98):
        print("\nReached 98% accuracy so cancelling training!")
        self.model.stop_training = True
        
# Instance of Callback
callB = myCallback()

# Load Data will retrun 4 lists of data.
# Testing and Training data test
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

"""
# In case we want to plot the image

import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(train_images[0])
print(train_labels[3])
print(train_images[3])

"""

# Normalizing data because it is easier to treat all values as either 0 or 1.
train_images  = train_images / 255.0
test_images = test_images / 255.0

# 3 Layers
model = keras.Sequential([
keras.layers.Flatten(input_shape=(28,28)), # Input Layer
keras.layers.Dense(128, activation = tf.nn.relu), # Hidden Layer
keras.layers.Dense(10,  activation = tf.nn.softmax) # Output Layer
])
print("Building\n")
# Building by compiling with an optimizer and a loss function
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Training\n")
# Then Train with Model.fit
model.fit(train_images, train_labels, epochs=15, callbacks = [callB])

print("Evaluate Model\n")
# Check how good the model is so far
model.evaluate(test_images,test_labels)

# Get Classification
classification = model.predict(test_images)
print(classification[0])

