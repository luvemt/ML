from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np



#Implementation of elbow approach
def elbow_approach(X,j=20):
	distort = []
	for k in range (1, j):
		km = KMeans(n_clusters=k)
		km.fit_predict(X)
		distort.append(km.inertia_)

	fig = plt.figure (figsize=(15,5))
	plt.plot(range(1,20), distort)
	plt.grid(True)
	plt.title('Elbow curve')
	plt.show()



def get_MinPts_and_E(X, path):

	if(path=="iris"):
		n= len(X[:,1])
	else:
		n = len(X.index)
	K=n-1
	nn=NearestNeighbors(n_neighbors=(K+1))
	nbrs = nn.fit(X)
	dists, indices = nbrs.kneighbors(X)

	distK = np.empty([K,n])

	for i in range(K):
		distK_i = dists[:,(i+1)]
		distK_i.sort()
		distK_i = distK_i[::-1]
		distK[i] = distK_i


	for i in range(15):
		plt.plot(distK[i], label='K=%d' %(i+1))
	plt.ylabel('distance')
	plt.xlabel('points')
	plt.legend()
	plt.show()

def ground_truth_accuracy_iris(y, cluster_labels):

	correctly_classified = 0
	for i in range(len(y)):
		if(y[i]== cluster_labels[i]):
			correctly_classified+=1
			
		elif (y[i]== cluster_labels[i]):
			correctly_classified+=1

		elif(y[i]==cluster_labels[i]):
			correctly_classified+=1

	return correctly_classified/len(y)



def std1(labels):
	new_labels=[]
	for i in range(len(labels)):
		if labels[i] == 1:
			new_labels.append(0)

		elif labels[i] == 0:
			new_labels.append(2) 

		elif labels[i] == 2:
			new_labels.append(1)

	return new_labels



def std2(labels):
	new_labels=[]
	for i in range(len(labels)):
		if labels[i] == 3:
			new_labels.append(0)

		elif labels[i] == 1:
			new_labels.append(2) 

		elif labels[i] == 2:
			new_labels.append(1)

	return new_labels
















