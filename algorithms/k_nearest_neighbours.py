import numpy as np
from collections import Counter
from distance_metrics import euclidean_distance

def k_nearest_neighbours(points, k, point, metric):
    sorted_points = sorted(points, key=lambda x: metric(point, x[0]))
    print(sorted_points[:k])
    labels = Counter([label for _, label in sorted_points[:k]])
    return labels.most_common(k)

points = [
    (np.array([[-2, 6]]), 1),
    (np.array([[-1,-4]]), 1),
    (np.array([[3,-1]]), 1),
    (np.array([[-3,-2]]), 2),
    (np.array([[-4,-5]]), 3)
    ]

point = np.array([[-2,0]])

print("returns as a counter - with the class label and number of occurances of the class label")

print("k nearest neighbours k=1 :")
print(k_nearest_neighbours(points,1,point, euclidean_distance))
print('\n')

print("k nearest neighbours k=3 :")
print(k_nearest_neighbours(points,3,point, euclidean_distance))
print('\n')

print("k nearest neighbours k=5 :")
print(k_nearest_neighbours(points,5,point, euclidean_distance))
print('\n')