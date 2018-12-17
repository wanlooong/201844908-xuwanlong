from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import mixture, metrics
from sklearn.cluster import *
import json
import numpy as np

texts = []
labels = []
data = open("Tweets.txt", 'r')
for line in data.readlines():
    content = json.loads(line)
    texts.append(content['text'])
    labels.append(content['cluster'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# K_Means
km = KMeans(n_clusters=100).fit(X)
km_pred = km.labels_
km_nmi = normalized_mutual_info_score(labels, km_pred)
print('K_Means:', km_nmi)

# Affinity Propagation
af = AffinityPropagation().fit(X)
af_pred = af.labels_
af_nmi = normalized_mutual_info_score(labels, af_pred)
print('Affinity Propagation:', af_nmi)

# Mean_Shift
ms = MeanShift(bandwidth=5).fit(X.todense())
ms_pred = ms.labels_
ms_nmi = normalized_mutual_info_score(labels, ms_pred)
print('Mean_Shift:', ms_nmi)

# Spectral Clustering
metrics_metrix = (-1 * metrics.pairwise.pairwise_distances(X)).astype(np.float32)
metrics_metrix += -1 * metrics_metrix.min()
sc_pred = spectral_clustering(metrics_metrix, n_clusters=100)
sc_nmi = normalized_mutual_info_score(labels, sc_pred)
print('Spectral Clustering:', sc_nmi)

# Ward Hierarchical Clustering
ward = AgglomerativeClustering(n_clusters=100, linkage='ward').fit(X.toarray())
ward_pred = ward.labels_
ward_nmi = normalized_mutual_info_score(labels, ward_pred)
print('Ward Hierarchical Clustering:', ward_nmi)

# Agglomerative Clustering
ac = AgglomerativeClustering(n_clusters=100).fit(X.toarray())
ac_pred = ac.labels_
ac_nmi = normalized_mutual_info_score(labels, ac_pred)
print('Agglomerative Clustering:', ac_nmi)

# DBSCAN
db = DBSCAN(eps=0.3, min_samples=1).fit(X.todense())
labels_predict = db.labels_
db_nml = normalized_mutual_info_score(labels, labels_predict)
print('DBSCAN:', db_nml)

# Gaussian Mixture
gmm = mixture.GaussianMixture(n_components=100, covariance_type='diag').fit(X.toarray())
gmm_pred = gmm.predict(X.toarray())
gmm_nmi = normalized_mutual_info_score(labels, gmm_pred)
print('Gaussian Mixture:', gmm_nmi)
