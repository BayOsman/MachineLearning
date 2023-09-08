import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
import cv2
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import seaborn as sns

%matplotlib inline
np.random.seed(1)
tf.random.set_seed(2)



label_path_normal = "/home/obayraktar/work/labeled/xnormal"
label_path_sick = "/home/obayraktar/work/labeled/patient"
main_path = "/home/obayraktar/work/labeled/"

size = 224
dataset =[]
labelset = []

normal_images = os.listdir(label_path_normal)
for i, image_name in enumerate(normal_images):
    if (image_name.split(".")[1] == "png"):
        os.chdir(label_path_normal)
        img = cv2.imread(image_name)
        img = cv2.resize(img,(size,size))
        dataset.append(np.array(img))
        labelset.append(0)
        os.chdir(main_path)

patient_images = os.listdir(label_path_sick)
for i, image_name in enumerate(patient_images):
    if (image_name.split(".")[1] == "png"):
        os.chdir(label_path_sick)
        img = cv2.imread(image_name)
        img = cv2.resize(img,(size,size))
        dataset.append(np.array(img))
        labelset.append(1)
        os.chdir(main_path)

dataset = np.array(dataset)
dummy_y = np_utils.to_categorical(labelset)

x_train, x_test, y_train, y_test = train_test_split(dataset,dummy_y,test_size=0.3)



x_train = x_train/255.
x_test = x_test/255.

def identity_block(X, f, filters, initializer=random_uniform):
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X) # Default axis
    X = Activation('relu')(X)
    
    ### START CODE HERE
    ## Second component of main path (≈3 lines)
    ## Set the padding = 'same'
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = "same", kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation("relu")(X)

    ## Third component of main path (≈2 lines)
    ## Set the padding = 'valid'
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = "valid", kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    
    ## Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X_shortcut,X])
    X = Activation("relu")(X)
    ### END CODE HERE

    return X

def convolutional_block(X, f, filters, s = 2, initializer=glorot_uniform):
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    
    # First component of main path glorot_uniform(seed=0)
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    ### START CODE HERE
    
    ## Second component of main path 
    X = Conv2D(filters = F2, kernel_size = f,strides = 1,padding = "same",kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X) 
    X = Activation("relu")(X)

    ## Third component of main path 
    X = Conv2D(filters = F3, kernel_size = 1, strides = 1, padding = "valid" , kernel_initializer = initializer(seed = 0))(X)
    X = BatchNormalization(axis = 3)(X) 
    
    ##### SHORTCUT PATH ##### 
    X_shortcut = Conv2D(filters = F3,kernel_size=1, strides = (s,s), padding = "valid", kernel_initializer = initializer(seed = 0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)
    
    ### END CODE HERE

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape = (size, size, 3), classes = 2, training=False):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    ### START CODE HERE
    
    # Use the instructions above in order to implement all of the Stages below
    # Make sure you don't miss adding any required parameter
    
    ## Stage 3 
    # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X,f = 3, filters = [128,128,512], s = 2)
    
    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X,3,[128,128,512])
    X = identity_block(X,3,[128,128,512])
    X = identity_block(X,3,[128,128,512])

    # Stage 4 
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, f = 3, filters = [256,256,1024], s=2)
    
    # the 5 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X,3,[256,256,1024])
    X = identity_block(X,3,[256,256,1024])
    X = identity_block(X,3,[256,256,1024])
    X = identity_block(X,3,[256,256,1024])
    X = identity_block(X,3,[256,256,1024])

    # Stage 5
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X,f=3, filters=[512,512,2048], s=2)
    
    # the 2 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X,3,[512,512,2048])
    X = identity_block(X,3,[512,512,2048])

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
    X = AveragePooling2D()(X)
    
    ### END CODE HERE

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model


model = ResNet50(input_shape = (size, size, 3), classes = 2)
np.random.seed(1)
tf.random.set_seed(2)
opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs = 100, batch_size = 16,validation_data=(x_test,y_test))


print ("Loss = " , history.history["loss"][-1])
print ("Test Accuracy = " , history.history["val_accuracy"][-1])


import matplotlib.pyplot as plt
prediction = np.argmax(model.predict(x_test),axis=1)
y_test1 = np.argmax(y_test,axis=1)
print(prediction)

plt.figure(figsize=(10,10))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(x_test[i])
    plt.ylabel("tahmin={}".format(prediction[i]))
    plt.title(np.argmax(y_test,axis=1)[i])


#loss graphs
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs =range(1,len(loss)+1)
plt.figure(13)
plt.plot(epochs,loss,"y",label="Training loss")
plt.plot(epochs,val_loss,"r",label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

#accuracy graphs
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs =range(1,len(acc)+1)
plt.figure(14)
plt.plot(epochs,acc,"y",label="Training acc")
plt.plot(epochs,val_acc,"r",label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# confusion matrix
cm = confusion_matrix(y_test1,prediction)
plt.figure(15),sns.heatmap(cm,annot=True,cmap="Blues")

#roc curve
fpr,tpr,threshold = roc_curve(y_test1,prediction)
plt.figure(16)
plt.plot([0,1],[0,1],"y--")
plt.plot(fpr,tpr,marker=".")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC CURVE")
plt.show()

