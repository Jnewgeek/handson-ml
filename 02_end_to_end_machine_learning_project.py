# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:36:25 2019

@author: HASEE

Chapter 02 end to end machine learning project
"""

#################################### Set UP ###################################
import os
os.chdir(os.getcwd())
import numpy as np
import pandas as pd
# set the random seed
np.random.seed(42)
# to plot pretty figure, set the figure params
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import matplotlib as mpl
mpl.rc("axes",labelsize=14)
mpl.rc("xtick",labelsize=12)
mpl.rc("ytick",labelsize=12)

# set the figure location
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

# define the function that save the figure
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# sklearn model
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
import time

################################## Get data ###################################
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#download data from internet and decompress it to csv file
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
#load the housing data
def load_housing_data(housing_path=HOUSING_PATH):
    if not os.path.exists(housing_path):
        fetch_housing_data()
    # load local file
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing=load_housing_data()
print(">> Load housing data sucessfully!\n")

## check out the data infomation and structure
#print("First 5 lines of data:\n"+"-"*40+"\n",housing.head())            # show the first 5 lines of data
#print("Data info:\n"+"-"*40+"\n")
#print(housing.info())            # show the data infomation, such as type and missing number
#print("Data description:\n"+"-"*40+"\n",housing.describe())        # show the data structure
#print("Ocean_proximity value_counts:\n"+"-"*40+"\n",housing.ocean_proximity.value_counts())   # show the ocean_proximity distribution
#
######################### plot the data distributon ############################
##plt.figure()
#housing.hist(bins=50,figsize=(20,15))
#save_fig("attribute_histogram_plots")
#
################### Split the data to train_data and test data #################
############## split_train_test_01
##def split_train_test(data,test_ratio=0.2,seed=42,random_=True):
##    if random_==True:
##        np.random.seed(seed)
##    # shuffle the data
##    shuffled_indices=np.random.permutation(len(data))
##    test_set_size=int(len(data)*test_ratio)
##    test_indices=shuffled_indices[:test_set_size]
##    train_indices=shuffled_indices[test_set_size:]
##    return data.iloc[train_indices],data.iloc[test_indices]
##
##train_set, test_set = split_train_test(housing)
##print("Random: ",len(train_set), "train +", len(test_set), "test")
##
############### split_train_test_02
### to make sure that works well when new data loaded
##from zlib import crc32
##
### to create a array that mark whether the index should be added to test data
##def test_set_check(identifier,test_ratio):
##    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
##
##def split_train_test_by_id(data,test_ratio,id_column):
##    ids=data[id_column]
##    in_test_set=ids.apply(lambda id_: test_set_check(id_, test_ratio))
##    return data.loc[~in_test_set],data.loc[in_test_set]
##
### by rows
##housing_with_id = housing.reset_index()   # adds an `index` column
##train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")  
##print("By rows: ",len(train_set), "train +", len(test_set), "test")
##
### by location
##housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
##train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
##print("By location: ",len(train_set), "train +", len(test_set), "test")
#
############## split_train_test_03
#
#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#print("By sklearn train_test_split: ",len(train_set), "train +", len(test_set), "test")
#
############ Stratified Sampling
##plot the median income
##housing["median_income"].hist()
##pd.cut()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# bar plot
#housing["income_cat"].value_counts().sort_index().plot(kind="bar")
#plt.xticks(rotation=0)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
print("By sklearn StratifiedShuffleSplit: ",len(strat_train_set), "train +", len(strat_test_set), "test")
#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
#print(housing["income_cat"].value_counts() / len(housing))

################ compare the error between the Stratified and random
#def income_cat_proportions(data):
#    return data["income_cat"].value_counts() / len(data)
#
#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#
#compare_props = pd.DataFrame({
#    "Overall": income_cat_proportions(housing),
#    "Random": income_cat_proportions(test_set),
#    "Stratified": income_cat_proportions(strat_test_set),
#}).sort_index()
#compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
#compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
#print(compare_props) 

#delete the extra column
try:
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
except:
    pass    
############## Discover and visualize the data to gain insights ###############
housing=strat_train_set.copy()
#housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
#
## set the scatter size and colors
#housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
#             s=housing.population/1000,label="population",
#             c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
#plt.legend()

# add map png
#import matplotlib.image as mpimg
#california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california.png')
#ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
#                       s=housing['population']/100, label="Population",
#                       c="median_house_value", cmap=plt.get_cmap("jet"),
#                       colorbar=False, alpha=0.4,
#                      )
#plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
#           cmap=plt.get_cmap("jet"))
#plt.ylabel("Latitude", fontsize=14)
#plt.xlabel("Longitude", fontsize=14)
#
#prices = housing["median_house_value"]
#tick_values = np.linspace(prices.min(), prices.max(), 11)
#cbar = plt.colorbar()
#cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
#cbar.set_label('Median House Value', fontsize=16)
#
#plt.legend(fontsize=16)
#save_fig("california_housing_prices_plot")
#
## find correction
#corr_matrix=housing.corr()
#print("Correction:\n"+"-"*40+"\n",corr_matrix["median_house_value"].sort_values(ascending=False))
#
## scatter matrix
#print(">> Plotting the scatter matrix of 4 variables...\n")
#from pandas.plotting import scatter_matrix
#
#attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
#scatter_matrix(housing[attributes],figsize=(12,8))
#save_fig("scatter_matrix_plot")
#
## plot scatter of median_house_value and median_income
#housing.plot(kind="scatter", x="median_income", y="median_house_value",
#             alpha=0.1)
#plt.axis([0, 16, 0, 550000])
#save_fig("income_vs_house_value_scatterplot")
#
## create different variable
#print(">> Create some new variables...")
#housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
#housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
#housing["population_per_household"]=housing["population"]/housing["households"]
#
## check out the new correction
#corr_matrix=housing.corr()
#print("New Correction:\n"+"-"*40+"\n",corr_matrix["median_house_value"].sort_values(ascending=False))

# Prepare the data for Machine Learning algorithms
housing=strat_train_set.drop("median_house_value",axis=1)
housing_label=strat_train_set["median_house_value"].copy()

# clean the missing data
#print(">> Replace the missing data with median number...\n")
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
#imputer=SimpleImputer(strategy="median")
## imputer can only caculate the median value on numeric data, so we should drop the text label 
housing_num=housing.drop("ocean_proximity",axis=1)
#imputer.fit(housing_num)
## 填补缺失值
#X=imputer.transform(housing_num)
#housing_tr=pd.DataFrame(X,columns=housing_num.columns)

# transform the character data
#print(">> Transform the Category to Number...\n")
#try:
#    from sklearn.preprocessing import OrdinalEncoder
#except ImportError:
#    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20
    
#housing_cat = housing[['ocean_proximity']]
#print("Raw label(first 10 lines):\n"+"-"*40+"\n",housing_cat.head(10))
#ordinal_encoder = OrdinalEncoder()
#housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
#print("Transformed label(first 10 lines):\n"+"-"*40+"\n",housing_cat_encoded[:10])
#print("Transformed Rules:\n"+"-"*40+"\n",ordinal_encoder.categories_)
#
## Create OneHotEncoder
#print(">> Create OneHotEncoder...\n")
try:
    #from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

#cat_encoder = OneHotEncoder()
#housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#print("One hot encoder:\n"+"-"*40+"\n",housing_cat_1hot.toarray())

# add extra feature
# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
#print(">> Add extra feature...\n")
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]
from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

#attr_adder = FunctionTransformer(add_extra_features, validate=False,
#                                 kw_args={"add_bedrooms_per_room": False})
#housing_extra_attribs = attr_adder.fit_transform(housing.values)
#housing_extra_attribs = pd.DataFrame(
#    housing_extra_attribs,
#    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
#    index=housing.index)
#print("After adding extra features:\n"+"-"*40+"\n",housing_extra_attribs.head())
#
## Create a pipeline to prepare the data
#print(">> Create a pipeline to prepare the data...\n")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20

# clean the numreric features
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
print("\n>> prepare housing data sucessfully!\n")

#################################################################################################
###################################### Select and train a model  ################################
#################################################################################################
def select_model(model='LR',housing_prepared=housing_prepared,housing_label=housing_label,load=True):
    '''select the LinearRegression,DecisionTreeRegressor or RandomForestRegressor.'''
    # save model
    from sklearn.externals import joblib
    # load the model if it exists
    import time
    time1=time.time()
    if model=="LR":
        from sklearn.linear_model import LinearRegression
        time1=time.time()
        model=LinearRegression()
        model_name="LinearRegression"
    elif model=="DT":
        from sklearn.tree import DecisionTreeRegressor
        model=DecisionTreeRegressor(random_state=42)
        model_name="DecisionTreeRegressor"
    elif model=="RF":
        from sklearn.ensemble import RandomForestRegressor
        model=RandomForestRegressor(n_estimators=10, random_state=42)
        model_name="RandomForestRegressor"
    elif model=="SVR":
        from sklearn.svm import SVR
        model = SVR(kernel="linear")
        model_name="SVR"
    else:
        return None
    if os.path.exists("model_set/%s.pkl"%model_name):
        model=joblib.load("model_set/%s.pkl"%model_name)
        print("\n>> Load < %s > model from the local sucessfully!\n"%model_name)
        return model
    # train the model
    model.fit(housing_prepared,housing_label)
    # caculate the RMSE
    from sklearn.metrics import mean_squared_error
    housing_predictions=model.predict(housing_prepared)
    mse=mean_squared_error(housing_label,housing_predictions)
    rmse=np.sqrt(mse)
    time2=time.time()
    print("%s trained sucessfully, use %.2fs, the rmse is %.6f."%(model_name,time2-time1,rmse))
    with open("model_set/model_statistics.txt",'a+',encoding="utf-8") as f:
        f.write("[ % s]"%time.ctime()+"%s trained sucessfully, use %.2fs, the rmse is %.6f."%(model_name,time2-time1,rmse))
    # Fine-tune your model
    from sklearn.model_selection import cross_val_score
    print("\n>> %s Scores:\n"%model_name+"-"*40+"\n")
    time1=time.time()
    scores=cross_val_score(model,housing_prepared,housing_label,
                           scoring="neg_mean_squared_error",cv=10)
    rmse_scores=np.sqrt(-scores)
    time2=time.time()
    # check out the final results
    def display_scores(scores,time_=time2-time1):
        print("scores:",scores)
        print("Mean:",scores.mean())
        print("Standard deviation:",scores.std())
        print("time used: %.2fs"%time_)
        with open("model_set/model_statistics.txt",'a+',encoding="utf-8") as f:
            f.write("scores: {}\n".format(scores))
            f.write("Mean: {}\n".format(scores.mean()))
            f.write("Standard deviation: {}\n".format(scores.std()))
            f.write("time used: %.2fs\n"%time_)
            f.write("-"*100+"\n")
    display_scores(rmse_scores)
    
    # save the model
    joblib.dump(model,"model_set/%s.pkl"%model_name)
    
    return model
    
## LinearRegression
#lin_reg=select_model()
#    
## DecisionTreeRegressor
#tree_reg=select_model("DT")
#
## RandomForestRegressor
#forest_reg=select_model("RF")
#
### SVR
#svr_reg=select_model("SVR")


#################################################################################################
###################################### Adjust the params smoothly ###############################
#################################################################################################
def find_best_model():
    #from sklearn.model_selection import GridSearchCV
    #import time
    #print("\n>> Starting Search the best params,please wait some seconds...")
    #time1=time.time()
    #
    #param_grid=[
    #        {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
    #        {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
    #        ]
    ## train the model 
    from sklearn.ensemble import RandomForestRegressor
    #forest_reg=RandomForestRegressor()
    #grid_search=GridSearchCV(forest_reg,param_grid,cv=5,
    #                         scoring='neg_mean_squared_error')
    #grid_search.fit(housing_prepared,housing_label)
    #time2=time.time()
    #print("\n>> Grid Search sucessfully,use time %.2fs\n"%(time2-time1))
    #print("-"*40)
    #print(grid_search.best_params_)
    #print(grid_search.best_estimator_)
    #print("-"*40)
    #cvres = grid_search.cv_results_
    #for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #    print(np.sqrt(-mean_score), params)
        
    # random to adjust the params
    import time
    print("\n>> Starting Search the best params randomly,please wait some seconds...")   
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    time1=time.time()
    
    param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_label)
    time2=time.time()
    print("\n>> Grid Search sucessfully,use time %.2fs\n"%(time2-time1))
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
        
    # show the importance of features
    feature_importances = rnd_search.best_estimator_.feature_importances_
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    #cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print("Importance of features:\n"+"-"*40+"\n")
    import pprint
    pprint.pprint(sorted(zip(feature_importances, attributes), reverse=True))
    return rnd_search

#################################################################################################
###################################### Final model ##############################################
#################################################################################################
def main():

    if os.path.exists("model_set/final_model.pkl"):
        final_model=joblib.load("model_set/final_model.pkl")
    else:
        from sklearn.metrics import mean_squared_error
        time1=time.time()
        rnd_search=find_best_model()
        final_model = rnd_search.best_estimator_
        
        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()
        
        X_test_prepared = full_pipeline.transform(X_test)
        final_predictions = final_model.predict(X_test_prepared)
        
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)
        time2=time.time()
        print("Final model finished,use time %.2fs,rmse is %.6f"%(time2-time1,final_rmse))
        
        # confidence interval
        from scipy import stats
        confidence = 0.95
        squared_errors = (final_predictions - y_test) ** 2
#        mean = squared_errors.mean()
        m = len(squared_errors)
        
        interval_array=np.sqrt(stats.t.interval(confidence, m - 1,
                                 loc=np.mean(squared_errors),
                                 scale=stats.sem(squared_errors)))
        print("95% confidence interval is",interval_array)
        
        # save model 
        
        joblib.dump(final_model,"model_set/final_model.pkl")
    
if __name__=="__main__":
    main()




















