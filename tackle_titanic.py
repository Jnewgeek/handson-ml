# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:19:39 2019

@author: Administrator

# Tackle The Titanic datasets
"""
import os
os.chdir(os.getcwd())

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc("axes",labelsize=14)
mpl.rc("xtick",labelsize=12)
mpl.rc("ytick",labelsize=12)
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
import seaborn as sns
sns.set(font="SimHei")

chapter_id="titanic"

def save_fig(fig_id,tight_layout=True):
    path=os.path.join(".","images",chapter_id,fig_id+".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format="png",dpi=300)
    
####################################### load data ###########################################
TITANIC_PATH = os.path.join("datasets", "titanic")
import pandas as pd
import time

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)

print(">> Starting loading data...")
time1=time.time()
train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")
time2=time.time()
print("finished! use time %.2fs."%(time2-time1))

#train_data.head()
#train_data.info()
#train_data.describe()
#train_data["Survived"].value_counts()
################################ Prepare the data ####################################
from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
def get_preprocess_pipeline(num_columns=["Age", "SibSp", "Parch", "Fare"],
                 cat_columns=["Pclass", "Sex", "Embarked"]):
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X[self.attribute_names]
        
    from sklearn.pipeline import Pipeline
    try:
        from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
    except ImportError:
        from sklearn.preprocessing import Imputer as SimpleImputer
    
    # 数值型数据取中位数填补缺失值
    #num_columns=["Age", "SibSp", "Parch", "Fare"]
    num_pipeline = Pipeline([
            ("select_numeric", DataFrameSelector(num_columns)),
            ("imputer", SimpleImputer(strategy="median")),
        ])
        
    #num_pipeline.fit_transform(train_data)
    
    # 字符型数据取众数填补缺失值
    class MostFrequentImputer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                            index=X.columns)
            return self
        def transform(self, X, y=None):
            return X.fillna(self.most_frequent_)
        
    try:
        from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
        from sklearn.preprocessing import OneHotEncoder
    except ImportError:
        from future_encoders import OneHotEncoder # Scikit-Learn < 0.20
        
    cat_pipeline = Pipeline([
            ("select_cat", DataFrameSelector(cat_columns)),
            ("imputer", MostFrequentImputer()),
            ("cat_encoder", OneHotEncoder(sparse=False)),
        ])
    #cat_pipeline.fit_transform(train_data)   
    
    # 合并特征
    from sklearn.pipeline import FeatureUnion
    preprocess_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
        ])
    return preprocess_pipeline
# prepared data finally
preprocess_pipeline=get_preprocess_pipeline()    
X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]

################################## Train model ######################################
def select_model(model_name="SVC",X_train=X_train,y_train=y_train):
    print(">> %s model...\n"%model_name+"-"*40)
    time.sleep(0.5)
    time1=time.time()
    if model_name=="SVC":
    # SVC 
        from sklearn.svm import SVC
        model = SVC(gamma="auto")
        #model.fit(X_train, y_train)
    elif model_name=="RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return None
    # cross_val_score
    from sklearn.model_selection import cross_val_score
    model_scores = cross_val_score(model, X_train, y_train, cv=10)
    time2=time.time()
    print("finished! use time %.2fs,%s mean score:"%(time2-time1,model_name),model_scores.mean())

    # test check
#    X_test = preprocess_pipeline.transform(test_data)
#    y_pred = svm_clf.predict(X_test)
    return model,model_scores

svm_clf,svm_scores=select_model()
forest_clf,forest_scores=select_model("RF")
def plot_modelScores():
    plt.figure(figsize=(8, 4))
    plt.plot([1]*10, svm_scores, ".")
    plt.plot([2]*10, forest_scores, ".")
    plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
    plt.ylabel("Accuracy", fontsize=14)
#plot_modelScores()

#################### add more feature
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
#train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
#train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()

# new pipeline
preprocess_pipeline=get_preprocess_pipeline(num_columns=["AgeBucket", "RelativesOnboard", "Fare"])
X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]

# new models
svm_clf,svm_scores=select_model("SVC",X_train,y_train)
forest_clf,forest_scores=select_model("RF",X_train,y_train)
plot_modelScores()


# Grid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
time1=time.time()

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
forest_reg = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='accuracy', random_state=42,
                                verbose=5,n_jobs=-1)
rnd_search.fit(X_train, y_train)
time2=time.time()
print("\n>> Grid Search sucessfully,use time %.2fs\n"%(time2-time1))
final_model=rnd_search.best_estimator_
# 预测值
test_data["AgeBucket"] = test_data["Age"] // 15 * 15
#train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
test_data["RelativesOnboard"] = test_data["SibSp"] + test_data["Parch"]
X_test_prepared = preprocess_pipeline.transform(test_data)
final_predictions = final_model.predict(X_test_prepared)


submission=load_titanic_data("gender_submission.csv")

# 混淆矩阵
from sklearn.metrics import confusion_matrix
true_survive=submission["Survived"].values
print("混淆矩阵:\n",confusion_matrix(true_survive,final_predictions))

from sklearn.metrics import precision_score, recall_score,f1_score
print("精确度:",precision_score(true_survive,final_predictions))
print("召回率:",recall_score(true_survive,final_predictions))
print("F1分数:",f1_score(true_survive,final_predictions))

# ROC
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(true_survive,final_predictions)
#
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
from sklearn.metrics import roc_auc_score
print("ROC值:",roc_auc_score(true_survive,final_predictions))

submission["Survived"]=final_predictions
submission.to_csv("./datasets/titanic/gender_submission_new.csv",index=False,encoding="utf-8")





