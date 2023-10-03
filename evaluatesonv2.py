import tensorflow as tf
import numpy as np
import cv2
from keras.utils import np_utils
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_curve,f1_score,precision_score,recall_score,accuracy_score,auc
import seaborn as sns
import matplotlib.pyplot as plt
import modelim
import evaluate




class evaluate():
    def __init__(self) -> None:
        pass

    def prediction_samples(x_test,y_test,prediction):
        plt.figure(figsize=(10,10))
        for i in range(12):
            plt.subplot(3,4,i+1)
            plt.imshow(x_test[i])
            plt.ylabel("tahmin={}".format(prediction[i]))
            plt.title(np.argmax(y_test,axis=1)[i])
        plt.show()

    def plot_loss_graph(loss,val_loss,epochs):
        plt.plot(epochs,loss,"b",label="Training loss")
        plt.plot(epochs,val_loss,"r",label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
        
    def plot_acc_graph(acc,val_acc,epochs):
        plt.plot(epochs,acc,"b",label="Training acc")
        plt.plot(epochs,val_acc,"r",label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()


    def confusion(y_test1,prediction):
        
        cm = confusion_matrix(y_test1,prediction,normalize="all")
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
        disp.plot(cmap=plt.cm.Blues, values_format=".2f")
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.show()

    def roc(y_test1,prediction):
        fpr,tpr,threshold = roc_curve(y_test1,prediction)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def performance_scores(loss,y_test1,prediction):
        f1 = f1_score(y_test1,prediction)
        pre = precision_score(y_test1,prediction)
        rec = recall_score(y_test1,prediction)
        loss = loss[-1]
        acc = accuracy_score(y_test1,prediction)

        print("Model performance parameter\nPrecision:{}\nRecall:{}\nF1-score:{}\nAccuracy:{}\nLoss:{}".format(pre,rec,f1,acc,loss))


    def equalize(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        rgb = cv2.cvtColor(equalized,cv2.COLOR_GRAY2BGR)
        return rgb


    def cross_roc(splits,size,opt,x_test,y_test):
        seed = 7
        np.random.seed(seed)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        i=1
        kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        

        for train, test in kfold.split(x_test, np.argmax(y_test,axis=1)):
            #compile and fit
            model = modelim.Resnet(size)
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(x_test[train],y_test[train],epochs=200, batch_size=16,verbose=0)
            
            #prediction
            y_pred_keras = np.argmax(model.predict(x_test[test]),axis=1)
            
            #roc curve plots
            fpr, tpr, thresholds = roc_curve(np.argmax(y_test[test],axis=1), y_pred_keras)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            i= i+1


        plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='blue',label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()


    def corss_loss_acc(splits,size,opt,x_train,y_train,x_test,y_test,epochs):
        seed = 7
        np.random.seed(seed)
        loses = []
        val_loses = []
        accs = []
        val_accs = []
        
        kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

        for train, test in kfold.split(x_train, np.argmax(y_train,axis=1)):
            model = modelim.Resnet(size)
            seed = 8
            #compile and fit
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            history = model.fit(x_train[train],y_train[train],epochs=200, batch_size=16,verbose=0,validation_data=(x_test,y_test),shuffle=True)
            loses.append(history.history["loss"])
            val_loses.append(history.history["val_loss"])
            accs.append(history.history["accuracy"])
            val_accs.append(history.history["val_accuracy"])
            
        mean_loses = np.sum(np.array(loses),axis=0)/splits
        mean_val_loses = np.sum(np.array(val_loses),axis=0)/splits
        mean_accs = np.sum(np.array(accs),axis=0)/splits
        mean_val_accs = np.sum(np.array(val_accs),axis=0)/splits
        
        
        plt.figure(figsize=(12, 4))
        plt.figure(10)
        plt.plot(epochs,mean_accs,"b",label="'Mean Training Accuracy'")
        plt.plot(epochs,mean_val_accs,"r",label="Mean Validation Accuracy")
        plt.title('Mean Learning Curves - Accuracy')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        
        plt.figure(11)
        plt.plot(epochs,mean_loses,"b",label="Mean Training Loss")
        plt.plot(epochs,mean_val_loses,"r",label="Mean Validation Loss")
        plt.title('Mean Learning Curves - Loss')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()