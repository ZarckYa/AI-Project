#Team number: 
#             Jufeng Yang    ID: 20125011
#             Xingda Zhou    ID: 19107471
#             Zhongen Qin    ID: 19107579
#Import the code libaries.
import tensorflow as tf
from tensorflow.keras import layers, datasets, Sequential, optimizers, utils
import numpy as np
import matplotlib.pyplot as plt


#Data preprocess.
#Load MNIST data into Variables
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data("MNIST_data")
for i in range(10):
    print('The %d train set label:'%(i),y_train[i])
    print('The %d test set label: '%(i),y_test[i])

#Reshape the data into [-1,28,28,1] form, and Normalize the data
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255
print("Train figures sets shape:", x_train.shape)
print("Test figures sets shape:", x_test.shape)
print("Train labels sets shape:", y_train.shape)
print("Test labels sets shape:", y_test.shape)

#To convert the label data into a matricx.
#This step can transfer 0 to [1,0,0,0,0,0,0,0,0,0]. like this format.
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
for i in range(10):
    print('The %d train set label:'%(i),y_train[i])
	print('The %d test set label: '%(i),y_test[i])



#Creat a CNN layers
Conv_layers = [
    #First layer
    layers.Conv2D(filters=32, kernel_size = [3,3], padding = 'same', input_shape = [28,28,1], activation = tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2]),
     #Second layer
    layers.Conv2D(filters=64, kernel_size = [3,3], padding = 'same', input_shape = [28,28,1], activation = tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2]),
    
    #Third layer
    layers.Conv2D(filters=128, kernel_size = [3,3], padding = 'same', input_shape = [28,28,1], activation = tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2]),
    
    #fLatten the all parameters
    layers.Flatten(),
    
    #Full connection layer with dropout 0.25.
    layers.Dense(128, activation = tf.nn.relu),
    layers.Dropout(0.25),
    layers.Dense(10, activation = tf.nn.softmax)
]


#Put those layer in to the Sequential list.
model = Sequential(Conv_layers)

#Output a report of the CNN layer.
model.summary()

#Training model

#The optimizer is Adam
#The loss function is categorical_crossentrop
#The matricx information is accuracy
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),loss = 'categorical_crossentropy',metrics = ['Accuracy'])

#Traning setting, define the epochs, batch_size. Save the hissory to a variable
History = model.fit(x_train,y_train, epochs = 10, batch_size = 500,validation_split = 0.1)

#Plot the training history in a diagram

#Plot the accuracy diagram
plt.figure(figsize = (8,8))
plt.subplot(1,2,1)
plt.plot(History.history['Accuracy'], label = 'Train') 
plt.plot(History.history['val_Accuracy'], label = 'Validation')
plt.title('The Training Accuracy History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')

#Plot the loss diagram
plt.subplot(1,2,2)
plt.plot(History.history['loss'], label = 'Train') 
plt.plot(History.history['val_loss'], label = 'Validation')
plt.title('The Training Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc = 'upper right')
plt.show()


#Test
Loss, accuracy = model.evaluate(x_test,y_test,batch_size=1,verbose=2)
#print the results out
print("The loss is: ", Loss)
print("The accuracy is :", accuracy)
