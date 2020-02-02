from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt
import datapreprocess
import optimal_params
import time
import sys


if len(sys.argv)!= 2:
	print ("Number of arguments should be 2")
	sys.exit()


X,y = datapreprocess.prepare_data(sys.argv[1])


#KMeans
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04,random_state=0)
start_time = time.time()
y_km =km.fit_predict((X))
total_time = time.time() - start_time
sse = km.inertia_
clustering_performance ="-"
if sys.argv[1]=="iris":
	clustering_performance = optimal_params.ground_truth_accuracy_iris(y,y_km)
print("\nKMEANS:")
print("SSE: ", sse)
print("Correctly classified(Relative to ground truth): ", clustering_performance)
print("Runtime: ", total_time)




#SciPy Hierarchical
start_time = time.time()
row_cluster= linkage(X, method='complete', metric='euclidean')
clusters = fcluster(row_cluster, 3, criterion='maxclust')
total_time = time.time() - start_time
clustering_performance ="-"
if sys.argv[1]=="iris":
	clustering_performance = optimal_params.ground_truth_accuracy_iris(y,y_km)
	clusters=optimal_params.std2(clusters)
print("\nSCIPY HIERARCHICAL:")
print("Correctly classified(Relative to ground truth): ", clustering_performance)
print("Runtime: ", total_time)


#Scikit-Learn Hierarchical
cl1 = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='complete')
start_time=time.time()
cl1_labels =cl1.fit_predict(X)
clustering_performance ="-"
if sys.argv[1]=="iris":
	clustering_performance = optimal_params.ground_truth_accuracy_iris(y,y_km)
	cl1_labels=optimal_params.std1(cl1_labels)
total_time = time.time() - start_time
print("\nSKLEARN HIERARCHICAL:")
print("Correctly classified(Relative to ground truth): ", clustering_performance)
print("Runtime: ", total_time)



#DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN (eps=0.7, min_samples=5, metric='euclidean')
start_time = time.time()
y_db = db.fit_predict(X)
clustering_performance ="-"
if sys.argv[1]=="iris":
	clustering_performance = optimal_params.ground_truth_accuracy_iris(y,y_km)
	clustering_performance = optimal_params.ground_truth_accuracy_iris(y,y_db)
total_time = time.time() - start_time
print("\nDBSCAN:")
print("Correctly classified(Relative to ground truth): ", clustering_performance)
print("Runtime: ", total_time)


#Elbow approach
optimal_params.elbow_approach(X)

#Get MinPts and E
optimal_params.get_MinPts_and_E(X,sys.argv[1])




