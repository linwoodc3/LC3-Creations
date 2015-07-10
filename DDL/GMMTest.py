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
from sklearn import mixture
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
print X_train.shape
v = mixture.VBGMM(n_components = 2, covariance_type = 'full')
v.fit(X_train)
v.predict(X_train)
v.score(X_test)

print v.score(X_test)

# display predicted scores by the model as a contour plot
x = np.linspace(-20.0, 40.0)
y = np.linspace(-20.0, 30.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z1 = v.score_samples(X_train)

CS = plt.contour(X, Y, Z1)
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
