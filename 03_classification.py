# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:51:19 2019

@author: Administrator

Chapter 03 Classification
"""
#################################### Set UP ###################################
from __future__ import division,print_function,unicode_literals

import os
import numpy as np

np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("axes",labelsize=14)
mpl.rc("xtick",labelsize=12)
mpl.rc("ytick",labelsize=12)
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

PROJECT_ROOT_DIR="."
CHAPTER_ID="classification"

def save_fig(fig_id,tight_layout=True):
    path=os.path.join(PROJECT_ROOT_DIR,"images",CHAPTER_ID,fig_id+".png")
    print("Saving figure",fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format="png",dpi=300)
    
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

################################## Get data ###################################
import time
import zipfile

def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]
    
def fetch_data(path):
    if not os.path.exists(path):
            os.mkdir(path)
    try:
        from sklearn.datasets import fetch_openml
        mnist=fetch_openml("mnist_784",version=1,cache=True)
        mnist.target=mnist.target.astype(np.int8)
        sort_by_target(mnist)
    except:
        from sklearn.datasets import fetch_mldata
        mnist=fetch_mldata("MNIST original")
    X=mnist["data"]
    y=mnist["target"]
    np.save(os.path.join(path,"mnist_X.npy"),X)
    np.save(os.path.join(path,"mnist_y.npy"),y)
    return X,y
    

def load_data():
    '''Load the mnist data from local or from the sklearn.datasets.'''
    print("\n>> Starting loading data,please wait some seconds...")
    time1=time.time()
    path=os.path.join(PROJECT_ROOT_DIR,"datasets","mnist")
    if os.path.exists(os.path.join(path,"mnist_X.zip")) and not os.path.exists(os.path.join(path,"mnist_X.npy")):
        for i in (os.path.join(path,"mnist_X.zip"),os.path.join(path,"mnist_y.zip")):
            housing_tgz = zipfile.ZipFile(i)
            housing_tgz.extractall(path=path)
            housing_tgz.close()
    if os.path.exists(os.path.join(path,"mnist_X.npy")):
        # 读取数据
        X=np.load(os.path.join(path,"mnist_X.npy"))
        y=np.load(os.path.join(path,"mnist_y.npy"))
        time2=time.time()
        print("\n>> Load MNIST data from local, use time %.2fs"%(time2-time1))
    else:
        X,y=fetch_data(path)
        time2=time.time()
        print("\n>> Download MNIST data from sklearn.datasets, use time %.2fs"%(time2-time1))
    return X,y

# get data
X,y=load_data()

############################# plot the mnist number ##########################################
#sample_data
some_digit = X[36000]

def plot_digit(data):
    image=data.reshape(28,28)
    plt.imshow(image,cmap=mpl.cm.binary,interpolation="nearest")
    plt.axis("off")
# plot the number 5
#plot_digit(some_digit)
# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    
#plt.figure(figsize=(9,9))
#example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
#plot_digits(example_images, images_per_row=10)
#save_fig("more_digits_plot")

#################################### Split the data ##########################################
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index=np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

del X;del y;del shuffle_index
###############################################################################################
###################################### 尝试阶段 ################################################
##############################################################################################

#################################### Binary classification ###################################
y_train_5=(y_train==5)
y_test_5=(y_test==5)
#
#from sklearn.linear_model import SGDClassifier
#
#sgd_clf=SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
#sgd_clf.fit(X_train,y_train_5)
#print("预测值:",sgd_clf.predict([some_digit]))
#
## cross_val_score
#from sklearn.model_selection import cross_val_score
#print("交叉验证:",cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy"))
#
#from sklearn.base import BaseEstimator
#
#class Never5Classifier(BaseEstimator):
#    def fit(self,X,y=None):
#        pass
#    def predict(self,X):
#        return np.zeros((len(X),1),dtype=bool)
#    
#never_5_clf=Never5Classifier()
#print("全部取负交叉验证:",cross_val_score(never_5_clf,X_train,y_train_5,cv=3,scoring="accuracy"))
#
#from sklearn.model_selection import cross_val_predict
#
#y_train_pred=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
#
## 混淆矩阵
#from sklearn.metrics import confusion_matrix
#print("混淆矩阵:",confusion_matrix(y_train_5,y_train_pred))
#
## 精确度和召回率
#from sklearn.metrics import precision_score, recall_score,f1_score
#
#print("精确度:",precision_score(y_train_5, y_train_pred))
#print("召回率:",recall_score(y_train_5,y_train_pred))
#print("F1分数:",f1_score(y_train_5,y_train_pred))
#
#y_scores=sgd_clf.decision_function([some_digit])
#print("阈值:",y_scores)
#
#from sklearn.metrics import precision_recall_curve
#y_scores=cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")
#precisions,recalls,thresholds=precision_recall_curve(y_train_5,y_scores)
#
############################ precision & recalls by thresholds ###############################
#def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
#    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
#    plt.xlabel("Threshold", fontsize=16)
#    plt.legend(loc="upper left", fontsize=16)
#    plt.ylim([0, 1])
#
#plt.figure(figsize=(8, 4))
#plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#plt.xlim([-700000, 700000])
#save_fig("precision_recall_vs_threshold_plot")
#
############################# Precision VS Recalls ############################################
#def plot_precision_vs_recall(precisions, recalls):
#    plt.plot(recalls, precisions, "b-", linewidth=2)
#    plt.xlabel("Recall", fontsize=16)
#    plt.ylabel("Precision", fontsize=16)
#    plt.axis([0, 1, 0, 1])
#
#plt.figure(figsize=(8, 6))
#plot_precision_vs_recall(precisions, recalls)
#save_fig("precision_vs_recall_plot")
#
##################################### ROC curve ###############################################
#from sklearn.metrics import roc_curve
#fpr,tpr,thresholds=roc_curve(y_train_5,y_scores)
#
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
#plt.figure(figsize=(8, 6))
#plot_roc_curve(fpr, tpr)
#save_fig("roc_curve_plot")
#
#from sklearn.metrics import roc_auc_score
#print("ROC值:",roc_auc_score(y_train_5,y_scores))
#
#################################### RandomForestClassifier ###################################
#from sklearn.ensemble import RandomForestClassifier
#forest_clf=RandomForestClassifier(n_estimators=10,random_state=42)
#y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
#                                    method="predict_proba")
#y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
#fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
#plt.figure(figsize=(8, 6))
#plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
#plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
#plt.legend(loc="lower right", fontsize=16)
#save_fig("roc_curve_comparison_plot")
#
## caculate the datas
#y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
#print("RandomForest:\n"+"-"*40)
#print("ROC值:",roc_auc_score(y_train_5, y_scores_forest))
#print("精确度:",precision_score(y_train_5, y_train_pred_forest))
#print("召回率:",recall_score(y_train_5,y_train_pred_forest))
#print("F1分数:",f1_score(y_train_5,y_train_pred_forest))    
#
################################## multiclass classification ##################################
#sgd_clf.fit(X_train, y_train)
#print("multiclass:",sgd_clf.predict([some_digit]))
#
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
#print("After Standarded:",cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
#
#y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
#conf_mx = confusion_matrix(y_train, y_train_pred)
#print("混淆矩阵:\n",conf_mx)
#
#def plot_confusion_matrix(matrix):
#    """If you prefer color and a colorbar"""
#    fig = plt.figure(figsize=(8,8))
#    ax = fig.add_subplot(111)
#    cax = ax.matshow(matrix)
#    fig.colorbar(cax)
#    
## 绘制混淆矩阵
#plot_confusion_matrix(conf_mx)
#save_fig("confusion_matrix_plot", tight_layout=False)
#
## 绘制错误率
#row_sums = conf_mx.sum(axis=1, keepdims=True)
#norm_conf_mx = conf_mx / row_sums
#np.fill_diagonal(norm_conf_mx, 0)
#plot_confusion_matrix(norm_conf_mx)
#save_fig("confusion_matrix_errors_plot", tight_layout=False)
#
## 3 和 5 的对照
#cl_a, cl_b = 3, 5
#X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
#X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
#X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
#X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
#
#plt.figure(figsize=(8,8))
#plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
#plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
#plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
#plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
#save_fig("error_analysis_digits_plot")
#
################################# Multilabel Classification ################################
#from sklearn.neighbors import KNeighborsClassifier
#
#y_train_large = (y_train >= 7)
#y_train_odd = (y_train % 2 == 1)
#y_multilabel = np.c_[y_train_large, y_train_odd]
#
#knn_clf = KNeighborsClassifier()
#knn_clf.fit(X_train, y_multilabel)
#print("Multilabel:",knn_clf.predict([some_digit]))
#
################################ Multioutput Classification ################################
#noise = np.random.randint(0, 100, (len(X_train), 784))
#X_train_mod = X_train + noise
#noise = np.random.randint(0, 100, (len(X_test), 784))
#X_test_mod = X_test + noise
#y_train_mod = X_train
#y_test_mod = X_test
#
#some_index = 5500
#plt.figure()
#plt.subplot(121); plot_digit(X_test_mod[some_index])
#plt.subplot(122); plot_digit(y_test_mod[some_index])
#save_fig("noisy_digit_example_plot")
#
#knn_clf.fit(X_train_mod, y_train_mod)
#clean_digit = knn_clf.predict([X_test_mod[some_index]])
#plt.figure()
#plot_digit(clean_digit)
#save_fig("cleaned_digit_example_plot")

#############################################################################################
########################################## 模型选取 ##########################################
#############################################################################################
# 随机猜测
#print(">> DummyClassifier:\n"+"-"*40)
#time.sleep(1)
#time1=time.time()
#from sklearn.dummy import DummyClassifier
#dmy_clf=DummyClassifier()
#y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")
#y_scores_dmy = y_probas_dmy[:, 1]
#fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)
#plot_roc_curve(fprr, tprr)
#time2=time.time()
#print("用时%.2fs,ROC值:"%(time2-time1),roc_auc_score(y_train_5,y_scores_dmy))
#
## kNN分类
#print(">> KNeighborsClassifier:\n"+"-"*40)
#time.sleep(1)
#time1=time.time()
#from sklearn.neighbors import KNeighborsClassifier
#knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
#knn_clf.fit(X_train, y_train)
#y_knn_pred = knn_clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#time2=time.time()
#print("用时%.2fs,准确度:"%(time2-time1),accuracy_score(y_test, y_knn_pred))
#
## 像素偏移
#from scipy.ndimage.interpolation import shift
#def shift_digit(digit_array, dx, dy, new=0):
#    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)
#
#plot_digit(shift_digit(some_digit, 5, 1, new=100))
#
## 数据增广
#print(">> 数据增广后的KNN分类:\n"+"-"*40)
#time.sleep(1)
#time1=time.time()
#X_train_expanded = [X_train]
#y_train_expanded = [y_train]
#for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
#    shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
#    X_train_expanded.append(shifted_images)
#    y_train_expanded.append(y_train)
#
#X_train_expanded = np.concatenate(X_train_expanded)
#y_train_expanded = np.concatenate(y_train_expanded)
#X_train_expanded.shape, y_train_expanded.shape
#
## 保存数据
#np.save("./datasets/mnist/mnists_X_expanded.npy",X_train_expanded)
#np.save("./datasets/mnist/mnists_y_expanded.npy",y_train_expanded)
#
#
#knn_clf.fit(X_train_expanded, y_train_expanded)
#y_knn_expanded_pred = knn_clf.predict(X_test)
#time2=time.time()
#print("用时%.2fs,准确度:"%(time2-time1),accuracy_score(y_test, y_knn_expanded_pred))
#
## 保存
##model_name="chapter03_knn"
##from sklearn.externals import joblib
##joblib.dump(knn_clf,"./model_set/%s.pkl"%model_name)
#
#ambiguous_digit = X_test[2589]
#knn_clf.predict_proba([ambiguous_digit])

###############################################################################################
################################## Exercise Solution ##########################################
###############################################################################################
# 1. An MNIST Classifier With Over 97% Accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_dic={"weights":["uniform","distance"],"n_neighbors":[3,4,5]}
knn_clf=KNeighborsClassifier()
grid_search=GridSearchCV(knn_clf,param_dic,cv=5,verbose=5,n_jobs=-1)
grid_search.fit(X_train,y_train) 

print("Best Params:",grid_search.best_params_)
print("Best Score:",grid_search.best_score_)

# test data
from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)
print("Test Accuracy:",accuracy_score(y_test, y_pred))

# 2. Data Augmentation
from scipy.ndimage.interpolation import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(X_train_augmented, y_train_augmented)

y_pred = knn_clf.predict(X_test)
print("After augmentation,Test accuracy:",accuracy_score(y_test, y_pred))

# 

















    
    
    
