from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import  AdaBoostClassifier,GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_y():
    y=[]
    y1 = [1] * 600 # malware
    y2 = [0] * 564# normal
    y = y1 + y2
    return y
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
def do_metrics(y_test,y_pred):
    print("metrics.accuracy_score:")
    print(metrics.accuracy_score(y_test, y_pred))
    print("metrics.confusion_matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("metrics.precision_score:")
    print(metrics.precision_score(y_test, y_pred))
    print("metrics.recall_score:")
    print(metrics.recall_score(y_test, y_pred))
    print("metrics.f1_score:")
    print(metrics.f1_score(y_test,y_pred))
# knn
def do_knn():
    data = pd.read_csv('data\\3_gram.csv')
    y = get_y()
    x = data.values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    print("knn:")
    y_pred = knn.predict(x_test)
    return y_pred
    do_metrics(y_test, y_pred)
# RandomForest
def do_rfc():

    data = pd.read_csv('data\\3_gram.csv')
    y = get_y()
    x = data.values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    print("RandomForest:")
    y_pred = rfc.predict(x_test)
    do_metrics(y_test, y_pred)
    return y_pred

# naive_bayes
def do_gnb():
    data = pd.read_csv('data\\3_gram.csv')
    y = get_y()
    x = data.values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print("naive_bayes:")
    y_pred = gnb.predict(x_test)
    return y_pred
    do_metrics(y_test, y_pred)

# AdaBoost
def do_adb(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    adb = AdaBoostClassifier(base_estimator=None, algorithm="SAMME", n_estimators=600, learning_rate=0.7,
                             random_state=None)
    adb.fit(x_train, y_train)
    print("AdaBoost:")
    y_pred = adb.predict(x_test)
    do_metrics(y_test, y_pred)
    #print(np.mean(model_selection.cross_val_score(adb, x, y, n_jobs=-1, cv=10)))
    return y_pred

#svm
def do_svm(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    clf_svm = svm.SVC()
    clf_svm.fit(x_train, y_train)
    print("svm:")
    y_pred = clf_svm.predict(x_test)
    do_metrics(y_test, y_pred)
    #print(np.mean(model_selection.cross_val_score(clf_svm, x, y, n_jobs=-1, cv=10)))
    return y_pred



#gbdt
def do_gbdt(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    clf_gbdt = GradientBoostingClassifier(random_state=10)
    print("gbdt:")
    clf_gbdt.fit(x_train,y_train)
    y_pred = clf_gbdt.predict(x_test)
    do_metrics(y_test, y_pred)
    print(np.mean(model_selection.cross_val_score(clf_gbdt, x, y, n_jobs=-1, cv=10)))
    return y_pred




if __name__ == '__main__':

    data = pd.read_csv('data\\3_gram.csv')
    y = get_y()
    x = data.values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

    f,ax=plt.subplots(2,2,figsize=(12,10))
    y_pred = do_rfc()
    sns.heatmap(confusion_matrix(y_test,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
    ax[0,0].set_title('Matrix for svm')

    y_pred = do_svm(x,y)
    sns.heatmap(confusion_matrix(y_test,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
    ax[0,1].set_title('Matrix for rfc')

    y_pred = do_adb(x,y)
    sns.heatmap(confusion_matrix(y_test,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
    ax[1,0].set_title('Matrix for adb')

    y_pred = do_gbdt(x,y)
    sns.heatmap(confusion_matrix(y_test, y_pred), ax=ax[1, 1], annot=True, fmt='2.0f')
    ax[1,1].set_title('Matrix for gbdt')

    plt.subplots_adjust(hspace=0.2,wspace=0.2)
    plt.savefig('D:\\Machine Learning\\2')
    plt.show()