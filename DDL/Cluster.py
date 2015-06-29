# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 

# useful links
# http://stackoverflow.com/questions/27504870/sklearn-kmeans-get-class-centroid-labels-and-reference-to-a-dataset  -> returns the class of the cluster; can see where each driver's trip fell

# June 12, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################


print(__doc__)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation


path = path = os.path.abspath(os.getcwd())

# Some colors for later
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)


###
 #load data from a CSV to a dataframe
with open("automatic.csv", 'ru') as in_data:
    skid_data = pd.DataFrame.from_csv(in_data, sep=',')

n_samples, n_features = skid_data.shape
print skid_data.shape

X = np.asfarray(skid_data[['Distance (mi)','Duration (min)','Fuel Cost (USD)','Average MPG','Fuel Volume (gal)','Hard Accelerations','Hard Brakes','Duration Over 70 mph (secs)',	'Duration Over 75 mph (secs)','Duration Over 80 mph (secs)']])

#number of groups
n_clusters=3

# Preprocessing tricks
#patched = StandardScaler().fit_transform(as_array)
#patched = scale(as_array, axis=0, with_mean=True)
patched = preprocessing.normalize(X, norm='l1')
#min_max_scaler = preprocessing.MinMaxScaler()
#patched = min_max_scaler.fit_transform(patched)
#print patched

'''
#Testing dimensionality reduction
pca = PCA(n_components=2)
small = pca.fit_transform(patched)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.noise_variance_) 
'''
svd = TruncatedSVD(n_components=2, n_iter = 100)
#reduction = svd.fit_transform(patched)

pca = PCA(n_components=2)
reduction = pca.fit_transform(patched)

# Compute Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

#print reduction
#print small


kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=20)
kmeans.fit_predict(reduction)

SilouetteCoefficient = metrics.silhouette_score(reduction, kmeans.labels_, metric='euclidean')

print "The Silhouette Coefficient score is \n>", SilouetteCoefficient


'''
n_clusters_ = len(cluster_centers_indices)

print X
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()



# array of indexes corresponding to classes around centroids, in the order of your dataset
classified_data = kmeans.labels_

#copy dataframe (may be memory intensive but just for illustration)
skid_data = skid_data.copy()
#print pd.Series(classified_data)
#print pd.Series(prediction_data)
skid_data['Cluster Class'] = pd.Series(classified_data, index=skid_data.index)
print skid_data.describe()
print skid_data
#print list(skid_data.columns)
skid_data.plot( x = 'Hard Accelerations', y = 'Cluster Class', kind = 'scatter')
plt.show()


# Silhouette Coefficient
print "We want scores close to 1 \n"

SilouetteCoefficient = metrics.silhouette_score(patched, classified_data, metric='euclidean')

print "The Silhouette Coefficient score is \n>", SilouetteCoefficient

AdjustRandIndex = metrics.adjusted_rand_score(classified_data, prediction_data)
MutualInfoScore = metrics.adjusted_mutual_info_score(classified_data,prediction_data)
HomogenietyScore = metrics.homogeneity_score(classified_data, prediction_data) 
CompletenessScore = metrics.completeness_score(classified_data, prediction_data)
V_measure = metrics.v_measure_score(classified_data, prediction_data) 


print "The Silouette Coefficient score is %r\nThe Adjusted Rand index is %r\nThe Mutual Information based score is %r\nThe Homogeneity score is %r\nThe completeness score is %r\nThe V-measure score is %r" % (SilouetteCoefficient,AdjustRandIndex,MutualInfoScore,HomogenietyScore,CompletenessScore,V_measure)
'''

#############
#scikit-learn visualization example

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduction[:, 0].min() + 1, reduction[:, 0].max() - 1
y_min, y_max = reduction[:, 1].min() + 1, reduction[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduction[:, 0], reduction[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the snippet of Team Skidmarks dataset \n(PCA-reduced data)'
          'Centroids are marked with blue cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
#plt.savefig('5clusterPCA.png', orientation = 'landscape')
plt.show()
#figsavepath = os.path.normpath(os.path.join(path,'figures',str(n_clusters)+"_cluster_KMeans_PCAReduced"+ ".png"))


