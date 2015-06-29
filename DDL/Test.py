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
# Visualization
###############################################################################

X['Cluster Class'] = pd.Series(kmeans.labels_, index=X.index)
X.plot( x = 'Hard Accelerations', y = 'Cluster Class', kind = 'scatter')
plt.show()

# Scatter plot of scores
# ~~~~~~~~~~~~~~~~~~~~~~
# 1) On diagonal plot X vs Y scores on each components
plt.figure(figsize=(12, 8))
plt.subplot(221)
X.plot( x = 'Hard Accelerations', y = 'Cluster Class', kind = 'scatter',"ob",label="Label")
plt.title('Hard Accelerations and Cluster Label)'
plt.legend(loc="best")

plt.subplot(224)
plt.plot(X_train_r[:, 1], Y_train_r[:, 1], "ob", label="train")
plt.plot(X_test_r[:, 1], Y_test_r[:, 1], "or", label="test")
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title('Comp. 2: X vs Y (test corr = %.2f)' %
          np.corrcoef(X_test_r[:, 1], Y_test_r[:, 1])[0, 1])
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")

# 2) Off diagonal plot components 1 vs 2 for X and Y
plt.subplot(222)
plt.plot(X_train_r[:, 0], X_train_r[:, 1], "*b", label="train")
plt.plot(X_test_r[:, 0], X_test_r[:, 1], "*r", label="test")
plt.xlabel("X comp. 1")
plt.ylabel("X comp. 2")
plt.title('X comp. 1 vs X comp. 2 (test corr = %.2f)'
          % np.corrcoef(X_test_r[:, 0], X_test_r[:, 1])[0, 1])
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())

plt.subplot(223)
plt.plot(Y_train_r[:, 0], Y_train_r[:, 1], "*b", label="train")
plt.plot(Y_test_r[:, 0], Y_test_r[:, 1], "*r", label="test")
plt.xlabel("Y comp. 1")
plt.ylabel("Y comp. 2")
plt.title('Y comp. 1 vs Y comp. 2 , (test corr = %.2f)'
          % np.corrcoef(Y_test_r[:, 0], Y_test_r[:, 1])[0, 1])
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())
plt.show()


