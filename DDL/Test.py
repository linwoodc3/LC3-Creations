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
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing


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
n_samples, n_features = X.shape
#Here we scale the features to a range of 
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
patched = min_max_scaler.fit_transform(Xnew) 
print "Patched is" , patched

###############################################################################
# Machine Learning Algorithm from Sci-kit Learn
###############################################################################

#Reducing the dimensions so we can visualize the telematic profile similarity
reduced_data = PCA(n_components=2).fit_transform(patched)
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
fit = kmeans.fit(reduced_data)
Z = kmeans.predict(reduced_data)

###############################################################################
# Evaluating cluster performance
###############################################################################

SilhouetteCoefficient = metrics.silhouette_score(patched, kmeans.labels_, metric='euclidean')



###############################################################################
# Visualization
###############################################################################

centers = kmeans.cluster_centers_
center_colors = colors[:len(centers)]
plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

#plt.subplot(1,4,idx+1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color=colors[Z].tolist(), s=10)
plt.title('K-means clustering on Ben\'s Automatic Plug-in dataset (PCA-reduced data)\n'
          'The Silhouette Coefficient is %s' %SilhouetteCoefficient)
plt.xticks(())
plt.yticks(())


plt.show()

###############################################################################
# Visualization of Classes and Features
###############################################################################

X['Cluster Class'] = pd.Series(kmeans.labels_, index=X.index)
X.plot( x = 'Hard Accelerations', y = 'Cluster Class', kind = 'scatter')
plt.show()

