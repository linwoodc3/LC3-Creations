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
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.text as txt
#from sklearn.cross_validation import train_test_split


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




###############################################################################
# Visualization of Classes and Features
###############################################################################

X['Cluster Class'] = pd.Series(kmeans.labels_, index=X.index)
X.plot( x = 'Hard Accelerations', y = 'Cluster Class', kind = 'scatter')


# row and column sharing
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')

ax1.plot(x = X['Hard Accelerations'], y = X['Cluster Class'], color = 'r')
ax1.set_title('Hard Acceleration')
ax1.set_xlabel('Number of Hard Accelerations')


ax2.scatter(X['Hard Brakes'], y = X['Cluster Class'], color = 'b')
ax2.set_title('Hard Brakes')
ax2.set_xlabel('# of Hard Brakes', )

ax3.scatter(X['Average MPG'], y = X['Cluster Class'],  color='g')
ax3.set_title('Average MPG')
ax3.set_xlabel('Average MPG')

ax4.plot(x=X['Duration Over 70 mph (secs)'], y = X['Cluster Class'], color = 'k')
ax4.set_title('Duration Over 70 mph')
ax4.set_xlabel('Time > 70mph')

ax5.plot(x=X['Duration (min)'], y = X['Cluster Class'], color = 'k')
ax5.set_title('Duration of trip (min)')
ax5.set_ylabel('Cluster Label')
ax5.set_xlabel('Duration')

ax6.plot(x=X['Distance (mi)'], y = X['Cluster Class'], color = 'k')
ax6.set_title('Distance of trip (mi)')
ax6.set_ylabel('Cluster Label')
ax6.set_xlabel('Distance')

plt.show()
