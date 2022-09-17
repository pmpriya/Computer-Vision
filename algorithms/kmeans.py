
import numpy as np
from distance_metrics import euclidean_distance


class KMeans:
    def __init__(self, samples, centroids, metric):
        self.samples = [np.array(sample) for sample in samples]
        self.centroids = [np.array(centroid) for centroid in centroids]
        self.metric = metric

    def learn(self):
        number_of_clusters = len(self.centroids)
        allotted_clusters = [0] * len(self.samples)

        has_changed = True
        print("> iteration: 1", )
        print("> centroids: ", self.get_centroids())
        iteration = 2

        while has_changed:

            has_changed = False

           #for i, sample in enumerate(self.samples):
           #     centroid_index = min(range(number_of_clusters), key=lambda a: self.metric(self.centroids[a], sample))
            #    if centroid_index != allotted_clusters[i]:
            #        has_changed = True
            #    allotted_clusters[i] = centroid_index

            for i, sample in enumerate(self.samples):
                dist = []
                for a in range(number_of_clusters):
                    dist.append(self.metric(self.centroids[a], sample))
                c_i = min(dist)
                centroid_index = 0
                for index, i in enumerate(dist):
                    if(i == c_i):
                        centroid_index = index
               #centroid_index = min(range(number_of_clusters), key=lambda a: self.metric(self.centroids[a], sample))
                if centroid_index != allotted_clusters[i]:
                    has_changed = True
                allotted_clusters[i] = centroid_index

            buckets = [[] for _ in range(number_of_clusters)]
            for feature_vector, cluster_index in zip(self.samples, allotted_clusters):
                buckets[cluster_index].append(feature_vector)
            for i in range(number_of_clusters):
                self.centroids[i] = np.mean(buckets[i], axis=0)
            print(buckets)
            del buckets

            print("> iteration: ", iteration)
            print("> centroids: ", self.get_centroids())
            iteration += 1

    def get_centroids(self):
        return self.centroids



samples = [(0,6), (7,5), (2,0) , (0,0), (10,10), (0,9)]
centroids = [(0,6), (7,5)]

model = KMeans(samples, centroids, euclidean_distance)
model.learn()
#print(model.get_centroids())