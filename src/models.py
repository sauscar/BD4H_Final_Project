from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn import metrics
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def logreg(X,y):
   
    k = 5
    kf = KFold(n_splits=k, random_state=None)
    lrm= LogisticRegression(solver= 'liblinear')
    
    roc_score = []
    acc_score = []
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.todense()[train_index,:],X.todense()[test_index,:]
        # [:10, :10]
        y_train , y_test = y[train_index] , y[test_index]
        
        lrm.fit(X_train,y_train)
        pred_values = lrm.predict(X_test)
        
        roc = metrics.roc_auc_score(y_test,pred_values )
        roc_score.append(round(roc,4))


    avg_roc_score = sum(roc_score) / k

    print("ROC of each fold - {}".format(roc_score))
    print("Avg ROC : {}".format(avg_roc_score))

    class_names = ['NEGATIVE', 'SEPSIS']

    resultMatrix = confusion_matrix(y_test,pred_values, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = resultMatrix, display_labels = class_names)
    disp = disp.plot(include_values = True, cmap = 'Blues', xticks_rotation=45)
    plt.show()


def lightgbm(X, y):

    k = 5
    kf = KFold(n_splits=k, random_state=None)
    lgbm = LGBMClassifier()

    roc_score = []
    acc_score = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.todense()[train_index, :], X.todense()[test_index, :]
        # [:10, :10]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        lgbm.fit(X_train, y_train)
        pred_values = lgbm.predict(X_test)

        roc = metrics.roc_auc_score(y_test, pred_values)
        avg = metrics.accuracy_score(y_test,pred_values)
        roc_score.append(roc)
        acc_score.append(avg)


    avg_roc_score = sum(roc_score) / k
    avg_accuracy_score = sum(acc_score)/k
    print("ROC of each fold - {}".format(roc_score))
    print("Avg ROC : {}".format(avg_roc_score))
    print("Accuracy of each fold - {}".format(acc_score))
    print("Avg Accuracy: {}".format(avg_accuracy_score))

    class_names = ['NEGATIVE', 'SEPSIS']

    resultMatrix = confusion_matrix(y_test,pred_values, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix = resultMatrix, display_labels = class_names)
    disp = disp.plot(include_values = True, cmap = 'Blues', xticks_rotation=45)
    plt.show()

