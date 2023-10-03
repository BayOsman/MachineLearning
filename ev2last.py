import tensorflow as tf
import numpy as np
import cv2
from keras.utils import np_utils
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,f1_score,precision_score,recall_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import modelim
import matplotlib.pyplot as plt
import os
import modelim
import evaluate
from PIL import Image

np.random.seed(1)
tf.random.set_seed(2)

label_path_normal = "/home/obayraktar/work/labeled/normal"
label_path_sick = "/home/obayraktar/work/labeled/patient"
target_path_normal = "/home/obayraktar/work/labeled/normal_aug"
target_path_sick = "/home/obayraktar/work/labeled/patient_aug"
main_path = "/home/obayraktar/work/labeled/"

size = 224
dataset =[]
labelset = []

normal_images = os.listdir(label_path_normal)
for i, image_name in enumerate(normal_images):
    if (image_name.split(".")[1] == "png"):
        os.chdir(label_path_normal)
        img = cv2.imread(image_name)
        #img = evaluate.evaluate.equalize(img)
        img = cv2.resize(img,(size,size))
        dataset.append(np.array(img))
        labelset.append(0)
        os.chdir(main_path)

patient_images = os.listdir(label_path_sick)
for i, image_name in enumerate(patient_images):
    if (image_name.split(".")[1] == "png"):
        os.chdir(label_path_sick)
        img = cv2.imread(image_name)
        #img = evaluate.evaluate.equalize(img)
        img = cv2.resize(img,(size,size))
        dataset.append(np.array(img))
        labelset.append(1)
        os.chdir(main_path)

aug_patient_images = os.listdir(target_path_sick)
for i, image_name in enumerate(aug_patient_images):
    if (image_name.split(".")[1] == "png"):
        os.chdir(target_path_sick)
        img = cv2.imread(image_name)
        #img = evaluate.evaluate.equalize(img)
        #img = img.reshape(img.shape[1],img.shape[2],-1)
        img = cv2.resize(img,(size,size))
        dataset.append(np.array(img))
        labelset.append(1)
        os.chdir(main_path)

aug_normal_images = os.listdir(target_path_normal)
for i, image_name in enumerate(aug_normal_images):
    if (image_name.split(".")[1] == "png"):
        os.chdir(target_path_normal)
        img = cv2.imread(image_name)
        #img = evaluate.evaluate.equalize(img)
        #img = img.reshape(img.shape[1],img.shape[2],-1)
        img = cv2.resize(img,(size,size))
        dataset.append(np.array(img))
        labelset.append(0)
        os.chdir(main_path)


dataset = np.array(dataset)
dummy_y = np_utils.to_categorical(labelset)

x_train, x_test, y_train, y_test = train_test_split(dataset,dummy_y,test_size=0.2)
#x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.25)

x_train = x_train/255.
x_test = x_test/255.
#x_val = x_val/255.

#model compile
model = modelim.Resnet(size)
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs = 200, batch_size = 16,validation_data=(x_test,y_test))

#loss and acc values
print ("Loss = " , history.history["loss"][-1])
print ("Validation Accuracy = " , history.history["val_accuracy"][-1])


prediction = np.argmax(model.predict(x_test),axis=1)
y_test1 = np.argmax(y_test,axis=1)
print(prediction)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs =range(1,len(loss)+1)
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs =range(1,len(acc)+1)


#prediction table
evaluate.evaluate.prediction_samples(x_test=x_test,y_test=y_test,prediction=prediction)

#loss_graph
evaluate.evaluate.plot_loss_graph(loss,val_loss,epochs)

#acc_graph
evaluate.evaluate.plot_acc_graph(acc,val_acc,epochs)

#confusion matrix
evaluate.evaluate.confusion(y_test1,prediction)

#confusion matrix v2
evaluate.evaluate.confusionv2(y_test1,prediction)

#roc
evaluate.evaluate.roc(y_test1,prediction)

#performance
evaluate.evaluate.performance_scores(loss,y_test1,prediction)

#k fold cross roc
evaluate.evaluate.cross_roc(5,size,opt,x_test=x_test,y_test=y_test)

#k fold graphs
evaluate.evaluate.corss_loss_acc(5,size,opt,x_train,y_train,x_test,y_test,epochs)

#augmentation normal
#evaluate.evaluate.augmentation_normal(label_path_normal,target_path_normal)

#augmentation patient
#evaluate.evaluate.augmentation_sick(label_path_sick,target_path_sick)