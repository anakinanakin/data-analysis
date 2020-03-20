import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def kmeansCal(centroid, X):
	#print (centroid)

	kmeans = KMeans(n_clusters = 3, init = centroid, n_init = 1, max_iter = 1)
	kmeans.fit(X)
	#get cluster labels
	labels = kmeans.predict(X)
	#get centroids values
	#final_centroids = kmeans.cluster_centers_

	#get all distances to all clusters
	alldistances = kmeans.fit_transform(X)

	print ("cluster of each instance:")
	print (labels)
	print ("distance of instances with centroids:")
	print (alldistances)
	print ("centroids coordinates:")
	print (kmeans.cluster_centers_)

	return kmeans, kmeans.cluster_centers_

#preprocessing data
def normalize(data):
	Max = max(data)
	Min = min(data)

	newData = []

	#cannot overwrite with v = (v - Min)/(Max - Min)
	for v in data:
		newData.append((v - Min)/(Max - Min))

	return newData

df = pd.read_csv("./FirstAssignmentBPI-FirstDataSet.csv");

compensation = df['Compensation_Amount'].values
case = df['Case_Duration(Hours)'].values
activities = df['Total_Activities'].values

first_centroids = [[20000, 1.5, 6], [8000, 0.07, 10], [30000, 4, 26]]

#normalize first_centroids
for l in first_centroids:
	l[0] = (l[0] - min(compensation))/(max(compensation) - min(compensation))
	l[1] = (l[1] - min(case))/(max(case) - min(case))
	l[2] = (l[2] - min(activities))/(max(activities) - min(activities))

init_centroids = np.ndarray(shape = (3,3), buffer = np.array(first_centroids))

#normalize df
compensation = normalize(compensation)
case = normalize(case)
activities = normalize(activities)

X = np.array(list(zip(compensation, case, activities)))

#repeat algorithm
print ("kmeans step 1:" )
kmeans, center = kmeansCal(init_centroids, X)
#print ("kmeans step 2:" )
#kmeans, center = kmeansCal(center, X)
#print ("kmeans step 3:" )
#kmeans, center = kmeansCal(center, X)