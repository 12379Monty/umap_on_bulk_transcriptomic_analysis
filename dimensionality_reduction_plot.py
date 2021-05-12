#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from openTSNE import TSNE
import umap

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import hdbscan
from sklearn import mixture

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from time import time
import seaborn as sns

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})



# filename = sys.argv[1]
# print(filename)
filename = 'GSE121239.csv'
data = pd.read_csv('~/data/' + filename)

label_group = data[(data.shape[0] - 12): (data.shape[0] - 11)] 

# remove label info
data_no_label = data[1:(data.shape[0]-12)]
data_no_label.drop(data_no_label.columns[[0,1]], axis = 1, inplace = True)
data_no_label = data_no_label.to_numpy()
data_no_label = np.transpose(data_no_label) 

label_group.drop(label_group.columns[[0,1]], axis = 1, inplace = True)
label_group = label_group.to_numpy()
label_group = label_group[0,:]
label_group = label_group.astype(int)


#######################################################################
# dimensionality reduction
# running time and plot
#######################################################################

# UMAP
t0 = time()
reducer = umap.UMAP(n_components = 2, random_state=143, n_jobs=-1, negative_gradient_method='bh')
embedding_umap = reducer.fit_transform(data_no_label)
t1 = time()
print('UMAP running time is: ' + str(t1-t0) + ' s' )

fig, ax = plt.subplots()
scatter = ax.scatter(embedding_umap[:, 0],embedding_umap[:, 1], c=[sns.color_palette(n_colors = 20)[x] for x in label_group] )
plt.axis('off')


# tSNE
t0 = time()
embedding_tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, negative_gradient_method='bh').fit(data_no_label)
t1 = time()
print('t-SNE running time is: ' + str(t1-t0) + ' s' )

fig, ax = plt.subplots()
scatter = ax.scatter(embedding_tsne[:, 0],embedding_tsne[:, 1], c=[sns.color_palette(n_colors = 20)[x] for x in label_group] )
plt.axis('off')

# MDS
t0 = time()
embedding_mds = MDS(n_components=2, n_jobs=-1).fit_transform(data_no_label )
t1 = time()
print('MDS running time is: ' + str(t1-t0) + ' s' )

fig, ax = plt.subplots()
scatter = ax.scatter(embedding_mds[:, 0],embedding_mds[:, 1], c=[sns.color_palette(n_colors = 20)[x] for x in label_group] )
plt.axis('off')

# PCA
t0 = time()
scaler = StandardScaler()
data_scale = scaler.fit_transform(data_no_label)
pca = PCA(n_components = 2)
embedding_pca = pca.fit_transform(data_scale)
t1 = time()
print('PCA running time is: ' + str(t1-t0) + ' s' )

fig, ax = plt.subplots()
scatter = ax.scatter(embedding_pca[:, 0],embedding_pca[:, 1], c=[sns.color_palette(n_colors = 20)[x] for x in label_group] )
plt.axis('off')
 
#######################################################################
# apply clustering algorithms to embedded low-dimensional space
#######################################################################

# set the number of clusters as the number of groups in data
num_clusters = 2

# K-means
km = KMeans(n_clusters = num_clusters, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
kmeans_umap = km.fit_predict(embedding_umap)
kmeans_tsne = km.fit_predict(embedding_tsne)
kmeans_mds = km.fit_predict(embedding_mds)
kmeans_pca = km.fit_predict(embedding_pca)

# Hierarchical clustering:  AgglomerativeClustering
clusterer_hierarchical = AgglomerativeClustering(n_clusters = num_clusters )
hc_umap = clusterer_hierarchical.fit_predict(embedding_umap)
hc_tsne = clusterer_hierarchical.fit_predict(embedding_tsne)
hc_mds = clusterer_hierarchical.fit_predict(embedding_mds)
hc_pca = clusterer_hierarchical.fit_predict(embedding_pca)

# SpectralClustering
spectral_Clustering = SpectralClustering(n_clusters = num_clusters, assign_labels="discretize", random_state=0)
spc_umap = spectral_Clustering.fit_predict(embedding_umap)
spc_tsne = spectral_Clustering.fit_predict(embedding_tsne)
spc_mds = spectral_Clustering.fit_predict(embedding_mds)
spc_pca = spectral_Clustering.fit_predict(embedding_pca)

# Gaussian mixture model
gmm = mixture.GaussianMixture(n_components = num_clusters, covariance_type='full')
gmm_umap = gmm.fit_predict(embedding_umap)
gmm_tsne = gmm.fit_predict(embedding_tsne)
gmm_mds = gmm.fit_predict(embedding_mds)
gmm_pca = gmm.fit_predict(embedding_pca)


# HDBSCAN
clusterer_hdbscan = hdbscan.HDBSCAN(min_cluster_size = 10, gen_min_span_tree=True)
hdbscan_umap = clusterer_hdbscan.fit_predict(embedding_umap)
hdbscan_tsne = clusterer_hdbscan.fit_predict(embedding_tsne)
hdbscan_mds = clusterer_hdbscan.fit_predict(embedding_mds)
hdbscan_pca = clusterer_hdbscan.fit_predict(embedding_pca)

#clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
#clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
#clusterer.condensed_tree_.plot()

cluster_result = [kmeans_umap,kmeans_tsne,kmeans_mds,kmeans_pca, 
                  hc_umap,hc_tsne,hc_mds,hc_pca,
                  spc_umap,spc_tsne,spc_mds,spc_pca,
                  gmm_umap,gmm_tsne,gmm_mds,gmm_pca,
                  hdbscan_umap,hdbscan_tsne,hdbscan_mds,hdbscan_pca]


file = open("clustering_accuracy_NMI.txt","a") #append mode 
file.write( filename + "\n") 
for _, cluster in enumerate(cluster_result): 
   nmi = normalized_mutual_info_score(label_group, cluster)
   file.write(str(nmi) + "\n")
file.close()

file = open("clustering_accuracy_ARI.txt","a") #append mode 
file.write( filename + "\n") 
for _, cluster in enumerate(cluster_result): 
   ari = adjusted_rand_score(label_group, cluster)
   file.write(str(ari) + "\n")
file.close()


#######################################################################
# Contour plot showing the visiting timestamp of each sample point
#######################################################################

label_days_cummulated = data[(data.shape[0] - 1): (data.shape[0])] 

label_days_cummulated.drop(label_days_cummulated.columns[[0,1]], axis = 1, inplace = True)
label_days_cummulated = label_days_cummulated.to_numpy()
label_days_cummulated = label_days_cummulated[0,:]
label_days_cummulated = label_days_cummulated.astype(int)

fig, ax = plt.subplots()
ax.tricontour(embedding_umap[20:,0], embedding_umap[20:,1], label_days_cummulated[20:], levels=15, linewidths=0.5, colors='k')
tcf =  ax.tricontourf(embedding_umap[20:,0], embedding_umap[20:,1], label_days_cummulated[20:], levels=50, cmap="OrRd")
fig.colorbar(tcf, ax=ax)
ax.plot(embedding_umap[20:,0], embedding_umap[20:,1], 'ko', ms=3)
plt.axis('off')


#######################################################################
# Varying the metric parameter by 'euclidean' (default), 'canberra', 'cosine'
#######################################################################


#for i in range(100, 150):
reducer = umap.UMAP(n_components = 2, metric='euclidean', random_state = 143)
embedding_umap = reducer.fit_transform(data_no_label)
fig, ax = plt.subplots()
clusterer_hierarchical = AgglomerativeClustering(n_clusters = 3)
clustering = clusterer_hierarchical.fit(embedding_umap[20:,])
ax.scatter(embedding_umap[20:, 0], embedding_umap[20:, 1], 
           c=[sns.color_palette(["#E41A1C", "#377EB8", "#4DAF4A"])[x] for x in clustering.labels_[:]])
plt.axis('off')


reducer = umap.UMAP(n_components = 2, metric='canberra', random_state = 143)
embedding_umap = reducer.fit_transform(data_no_label)
fig, ax = plt.subplots()
clusterer_hierarchical = AgglomerativeClustering(n_clusters = 3)
clustering = clusterer_hierarchical.fit(embedding_umap[20:,])
ax.scatter(embedding_umap[20:, 0], embedding_umap[20:, 1], 
           c=[sns.color_palette(["#E41A1C", "#377EB8", "#4DAF4A"])[x] for x in clustering.labels_[:]])
plt.axis('off')

reducer = umap.UMAP(n_components = 2, metric='cosine', random_state = 143)
embedding_umap = reducer.fit_transform(data_no_label)
fig, ax = plt.subplots()
clusterer_hierarchical = AgglomerativeClustering(n_clusters = 3)
clustering = clusterer_hierarchical.fit(embedding_umap[20:,])
ax.scatter(embedding_umap[20:, 0], embedding_umap[20:, 1], 
           c=[sns.color_palette(["#E41A1C", "#377EB8", "#4DAF4A"])[x] for x in clustering.labels_[:]])
plt.axis('off')



# Hierarchical clustering:  AgglomerativeClustering
reducer = umap.UMAP(n_components = 2, metric='canberra', random_state = 143)
embedding_umap = reducer.fit_transform(data_no_label)
fig, ax = plt.subplots()
clusterer_hierarchical = AgglomerativeClustering(n_clusters = 3)
clustering = clusterer_hierarchical.fit(embedding_umap[20:,])
ax.scatter(embedding_umap[20:, 0], embedding_umap[20:, 1], 
           c=[sns.color_palette(["#E41A1C", "#377EB8", "#4DAF4A"])[x] for x in clustering.labels_[:]])
plt.axis('off')

# up    
# for p_id in  [1409, 1927,1938,2067,1178,1944,699,1478,1620,2016,1520,
#              113,759,1913, 1764,1176,46,704,1869,2128,981]:
# down
# for p_id in [1537,1792,1924,1424,1182,1699,1705,1206,1335,2103,2104,317,453,1480,
#                 1871,2129,725,2132,365,1263,371,244,911,1679,1807]:
# mixed
# for p_id in [1041,1174,24,1179,1052,1436,1702,1842,1463,966,458,1227,
#                2122,345,2020,1001,1269, 2119,582]:

patient_id = data[(data.shape[0] - 7): (data.shape[0] - 6)] 
patient_id.drop(patient_id.columns[[0,1]], axis = 1, inplace = True)
patient_id = patient_id.to_numpy()
patient_id = patient_id[0,:]
patient_id = patient_id.astype(int)

for p_id in [1041,1174,24,1179,1052,1436,1702,1842,1463,966,458,1227,
                2122,345,2020,1001,1269, 2119,582]:
    patient_visit = embedding_umap[patient_id == p_id]
    for pos in range(0, len(patient_visit)-1 ):
        x_pos = patient_visit[pos][0]
        y_pos = patient_visit[pos][1]
        x_direct = patient_visit[pos+1][0] - patient_visit[pos][0]
        y_direct = patient_visit[pos+1][1] - patient_visit[pos][1]
        ax.quiver(x_pos, y_pos, x_direct, y_direct, angles='xy', scale_units='xy', scale=1,
                   headwidth=2, linewidths = 1., alpha = 0.85,
                   color = sns.color_palette("Reds", 8)[pos])
plt.axis('off')
plt.show()






