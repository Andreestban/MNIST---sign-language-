import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing datasets
test_set = pd.read_csv("sign_mnist_test.csv")
training_set = pd.read_csv("sign_mnist_train.csv")

#seprrating dataset for training set and test set

#for training set ,taking label in y
training_set_X = training_set.iloc[:,1:].values
training_set_y = training_set.iloc[:,0].values

#for training set ,taking label in y
test_set_X = test_set.iloc[:,1:].values
test_set_y = test_set.iloc[:,0].values

#chaning shape of training and test data
training_set_X = training_set_X.reshape(training_set_X.shape[0],28,28,1)
test_set_X = test_set_X.reshape(test_set_X.shape[0],28,28,1)

#change data to float
training_set_X = training_set_X.astype('float32')
test_set_X = test_set_X.astype('float32')

#scale data
training_set_X = training_set_X/255
test_set_X = test_set_X/255

from keras.utils import to_categorical
training_set_y = to_categorical(training_set_y)
test_set_y = to_categorical(test_set_y)

#applying convolutional neural network

#importing libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#initailizing model
model = Sequential()

#applying 1st convolutional layer
model.add(Conv2D(64,(3,3),
                 activation = 'relu',
                 input_shape = (28,28,1)))

#applying maxpooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

#adding droput layer
model.add(Dropout(0.25))

#applying 2nd convolutional layer
model.add(Conv2D(64,(3,3),
                 activation = 'relu'))

#applying maxpooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

#applying dropout layer
model.add(Dropout(0.50))

#applying flatten layer
model.add(Flatten())

#applying classic Artificial neural networks
model.add(Dense(units = 128,
                activation = 'relu'))
model.add(Dense(units = 25,
                activation = 'softmax'))

#compiling model
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#fitting model
#saving model model for graph puposes in " history " variable
history = model.fit(training_set_X,training_set_y,
          batch_size = 128,
          epochs = 10,
          validation_data = (training_set_X,training_set_y))

#evaluating model
model.evaluate(test_set_X,test_set_y)

#plotting graph

# Plot train & test accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot train & test loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#plotting model
from keras.utils import plot_model
plot_model(model, to_file = 'Model_architecture.png')

model.save("Model_mnist_signLanguage.h5")