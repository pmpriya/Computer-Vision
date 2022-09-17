import numpy as np


class LDA:
    def __init__(self, xt, labels, weights):

        print("for weights: ", weights)
        for x in xt:
            y = np.dot(weights,x.transpose())
            print("> x: ", x)
            print("> y = wx: ", y)

        print('\n')

        m1 = []
        m2 = []
        sw = 0

        for i in range(0, len(xt)):
            if(labels[i] == 1):
                m1.append(xt[i])
            else:
                m2.append(xt[i])

        mean1 = np.mean(np.array(m1), axis=0)
        mean2 = np.mean(np.array(m2), axis=0)

        sb = (abs(np.dot(weights,((mean1-mean2).transpose()))))**2

        for x in m1:
            sw += (np.dot(weights,(x-mean1).transpose()))**2

        for x in m2:
            sw += (np.dot(weights,(x-mean2).transpose()))**2

        cost = sb/sw
        print("> Weights: ", weights)
        print("> mean 1: ", mean1)
        print("> mean 2: ", mean2)
        print("> sb: ", sb)
        print("> sw: ", sw)
        print("> Cost: ", cost.round(4))
        print('\n')
        print('\n')

def fishers_method_cost(labelled_samples, weights):
    cluster_points = {}
    for sample, label in labelled_samples:
        if label not in cluster_points:
            cluster_points[label] = []
        cluster_points[label].append(sample)
    cluster_points = cluster_points.values()

    centroids = []
    for cluster in cluster_points:
        centroids.append(np.mean(cluster, axis=0))

    sb = 0
    for i, centroid_a in enumerate(centroids):
        for centroid_b in centroids[i:]:
            sb += np.matmul(weights, centroid_a - centroid_b)**2

    sw = 0
    for i, cluster in enumerate(cluster_points):
        centroid = centroids[i]
        for point in cluster:
            sw += np.matmul(weights, centroid - point)**2

    cost = sb/sw

    print("Between class scatter (sb): %f" % sb)
    print("Within class scatter (sw): %f" % sw)
    print("Cost: %f" % cost)

    return cost


# HIGHER THE COST, MORE EFFECTIVE THE PROJECTION WEIGHT
xt = np.array([
    [1,2],
    [2,1],
    [3,3],
    [6,5],
    [7,8]
])

labels = np.array([1,1,1,2,2])

weights = [-1,5]
LDA(xt,labels,weights)
print("HIGHER THE COST, MORE EFFECTIVE THE PROJECTION WEIGHT")
print('\n')

weights2 = [2,-3]
LDA(xt,labels,weights2)

