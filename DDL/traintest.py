# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 

# District Data Labs Blog Post Data Analysis; Driver Telematics

# June 27, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################

import os
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cluster import KMeans
from sklearn import metrics, grid_search
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import make_scorer, silhouette_score
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn import preprocessing
import matplotlib.text as txt
from sklearn.cross_validation import train_test_split, cross_val_score, KFold



###############################################################################
# Administrative and Data Loading and Munging
###############################################################################

# Some colors for later
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)


#load data from a CSV to a dataframe
with open("automatic.csv", 'ru') as in_data:
    X = pd.DataFrame.from_csv(in_data, sep=',')

#Turn data into a numpy array for machine learning
Xnew = np.asfarray(X[['Distance (mi)','Duration (min)','Fuel Cost (USD)','Average MPG','Fuel Volume (gal)','Hard Accelerations','Hard Brakes','Duration Over 70 mph (secs)',	'Duration Over 75 mph (secs)','Duration Over 80 mph (secs)']])


X_train, X_test = train_test_split(Xnew, test_size = .5)
n_clusters, n_features = X_train.shape
silscore = make_scorer(silhouette_score, greater_is_better =True)
parameters = {'n_clusters':[2,100], 'max_iter':[300, 600], 'n_init':[1,50], 'init':('k-means++', 'random'), 'precompute_distances':('auto',True,False), 'tol':[0.00005,100], 'verbose':[0,1000]}
kmeans = KMeans()
clf = grid_search.GridSearchCV(kmeans, parameters, scoring = silscore)
clf.fit(X_train)
clf.score(X_test)
clf.get_params(deep=True)

print(clf.best_params_)
