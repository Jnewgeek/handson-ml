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
print("Load housing data sucessfully!")

# check out the data infomation and structure
print(housing.head())            # show the first 5 lines of data
print(housing.info())            # show the data infomation, such as type and missing number
print(housing.describe())        # show the data structure
print(housing.ocean_proximity.value_counts())   # show the ocean_proximity distribution

######################## plot the data distributon ############################
#plt.figure()
housing.hist(bins=50,figsize=(20,15))
save_fig("attribute_histogram_plots")

################## Split the data to train_data and test data #################
############# split_train_test_01
#def split_train_test(data,test_ratio=0.2,seed=42,random_=True):
#    if random_==True:
#        np.random.seed(seed)
#    # shuffle the data
#    shuffled_indices=np.random.permutation(len(data))
#    test_set_size=int(len(data)*test_ratio)
#    test_indices=shuffled_indices[:test_set_size]
#    train_indices=shuffled_indices[test_set_size:]
#    return data.iloc[train_indices],data.iloc[test_indices]
#
#train_set, test_set = split_train_test(housing)
#print("Random: ",len(train_set), "train +", len(test_set), "test")
#
############## split_train_test_02
## to make sure that works well when new data loaded
#from zlib import crc32
#
## to create a array that mark whether the index should be added to test data
#def test_set_check(identifier,test_ratio):
#    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
#
#def split_train_test_by_id(data,test_ratio,id_column):
#    ids=data[id_column]
#    in_test_set=ids.apply(lambda id_: test_set_check(id_, test_ratio))
#    return data.loc[~in_test_set],data.loc[in_test_set]
#
## by rows
#housing_with_id = housing.reset_index()   # adds an `index` column
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")  
#print("By rows: ",len(train_set), "train +", len(test_set), "test")
#
## by location
#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
#print("By location: ",len(train_set), "train +", len(test_set), "test")

############# split_train_test_03
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("By sklearn train_test_split: ",len(train_set), "train +", len(test_set), "test")

########### Stratified Sampling
#plot the median income
#housing["median_income"].hist()
#pd.cut()
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# bar plot
#housing["income_cat"].value_counts().sort_index().plot(kind="bar")
#plt.xticks(rotation=0)
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
print("By sklearn StratifiedShuffleSplit: ",len(train_set), "train +", len(test_set), "test")
#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
#print(housing["income_cat"].value_counts() / len(housing))

################ compare the error between the Stratified and random
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Random": income_cat_proportions(test_set),
    "Stratified": income_cat_proportions(strat_test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
print(compare_props) 

#delete the extra column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
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
import matplotlib.image as mpimg
california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
 
























