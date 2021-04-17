import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split


def logreg(X, y):

    k = 5
    kf = KFold(n_splits=k, random_state=None)
    lrm = LogisticRegression(solver="liblinear")

    roc_score = []

    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.todense()[train_index, :], X.todense()[test_index, :]
        # [:10, :10]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        lrm.fit(X_train, y_train)
        pred_values = lrm.predict(X_test)

        roc = metrics.roc_auc_score(pred_values, y_test)
        roc_score.append(roc)

    avg_roc_score = sum(roc_score) / k

    print("Accuracy of each fold - {}".format(roc_score))
    print("Avg accuracy : {}".format(avg_roc_score))


def lightgbm(X, y):

    k = 5
    kf = KFold(n_splits=k, random_state=None)
    lgbm = LGBMClassifier()

    roc_score = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.todense()[train_index, :], X.todense()[test_index, :]
        # [:10, :10]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        lgbm.fit(X_train, y_train)
        pred_values = lgbm.predict(X_test)

        roc = metrics.roc_auc_score(pred_values, y_test)
        roc_score.append(roc)

    avg_roc_score = sum(roc_score) / k
    print("Accuracy of each fold - {}".format(roc_score))
    print("Avg accuracy : {}".format(avg_roc_score))
    # print(metrics.roc_auc_score(y_test, y_pred))

