# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:42:25 2019

@author: Administrator

setup.py
"""

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

def save_fig(fig_id,CHAPTER_ID,tight_layout=True):
    path=os.path.join(PROJECT_ROOT_DIR,"images",CHAPTER_ID,fig_id+".png")
    print("Saving figure",fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format="png",dpi=300)

