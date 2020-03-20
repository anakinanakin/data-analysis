import pandas as pd
import numpy as np
import math

#calculate a list of distances between instances and centroids
def distanceCal(values, centroid, num):
	l1 = []
	l2 = []
	l3 = []

	for v in values:
		l1.append(v - centroid[0][num])
		l2.append(v - centroid[1][num])
		l3.append(v - centroid[2][num])

	l = [l1, l2, l3]

	return l

#calculate euclidean distances
def euclideanCal(compensation, case, activities, num):
	distances = []

	for idx, val in enumerate(compensation[0]):
		eucliDistance = math.sqrt(compensation[num][idx]**2 + case[num][idx]**2 + activities[num][idx]**2)
		distances.append(eucliDistance)

	return distances

#single kmeans algotithm
def kmeansCal(centroid, compensation, case, activities):
	comp_distances = distanceCal(compensation, centroid, 0)
	case_distances = distanceCal(case, centroid, 1)
	activ_distances = distanceCal(activities, centroid, 2)

	#print(comp_distances, case_distances, activ_distances)

	#euclidean distance of instances to centroids
	distance1 = euclideanCal(comp_distances, case_distances, activ_distances, 0)
	distance2 = euclideanCal(comp_distances, case_distances, activ_distances, 1)
	distance3 = euclideanCal(comp_distances, case_distances, activ_distances, 2)

	#cluster label
	label = []

	#specify each cluster instances to calculate new centroid
	ctr = 0
	cluster1 = []
	cluster2 = []
	cluster3 = []

	distances = np.array([distance1, distance2, distance3])
	distances = np.transpose(distances)

	#get cluster label of each instance
	for idx, val in enumerate(distance1):
		Min = min(distance1[idx], distance2[idx], distance3[idx])
		if Min == distance1[idx]:
			label.append(1)
			cluster1.append(ctr)
		elif Min == distance2[idx]:
			label.append(2)
			cluster2.append(ctr)
		else:
			label.append(3)
			cluster3.append(ctr)
		ctr+=1

	#new centroids
	centroid1 = newCentroid(cluster1, compensation, case, activities)
	centroid2 = newCentroid(cluster2, compensation, case, activities)
	centroid3 = newCentroid(cluster3, compensation, case, activities)
	centroid = [centroid1, centroid2, centroid3]

	print("cluster of each instance:")
	print(label)

	print("distance of instances with centroids:")
	print(distances)

	return centroid

#calculate new centroids
def newCentroid(cluster, compensation, case, activities):
	compensationSum = 0
	caseSum = 0
	activitiesSum = 0

	for v in cluster:
		compensationSum += compensation[v]
		caseSum += case[v]
		activitiesSum += activities[v]

	centroid = [compensationSum/len(cluster), caseSum/len(cluster), activitiesSum/len(cluster)]

	return centroid

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

first_centroids = [[20000, 1.5, 6], [8000, 0.07, 10], [30000, 4, 26]]

compensation = df['Compensation_Amount'].values
case = df['Case_Duration(Hours)'].values
activities = df['Total_Activities'].values

#normalize first_centroids
for l in first_centroids:
	l[0] = (l[0] - min(compensation))/(max(compensation) - min(compensation))
	l[1] = (l[1] - min(case))/(max(case) - min(case))
	l[2] = (l[2] - min(activities))/(max(activities) - min(activities))

#normalize df
compensation = normalize(compensation)
case = normalize(case)
activities = normalize(activities)

#print(first_centroids, compensation, case, activities)

#repeat algorithm
print("kmeans step 1:" )
first_centroids = kmeansCal(first_centroids, compensation, case, activities)
print("kmeans step 2:" )
first_centroids = kmeansCal(first_centroids, compensation, case, activities)
print("kmeans step 3:" )
first_centroids = kmeansCal(first_centroids, compensation, case, activities)