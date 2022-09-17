import numpy as np
from distance_metrics import euclidean_distance


class LeaderFollowerAlgorithm:
    def __init__(self, samples, theta, n, selection_order):
        self.samples = [np.array(sample) for sample in samples]
        self.centroids = []
        self.theta = theta
        self.n = n
        self.selection_order = selection_order

    def index_of_nearest_cluster(self, sample):
        number_of_clusters = len(self.centroids)
        return min(range(number_of_clusters), key=lambda i: euclidean_distance(self.centroids[i], sample))

    def learn(self):
        for index in self.selection_order:
            sample = self.samples[index]

            if len(self.centroids) == 0:
                self.centroids.append(sample)

            else:
                j = self.index_of_nearest_cluster(sample)
                if euclidean_distance(sample, self.centroids[j]) < self.theta:
                    self.centroids[j] = self.centroids[j] + self.n * (sample - self.centroids[j])
                else:
                    self.centroids.append(sample)

    def get_centroids(self):
        print("> centroids: ")
        print(self.centroids)
        return self.centroids

    def classify_sample(self, sample):
        centroid_index = self.index_of_nearest_cluster(sample)
        return self.centroids[centroid_index]

samples = [(-1, 3), (1, 4), (0, 5), (4, -1), (3, 0), (5, 1)]
model = LeaderFollowerAlgorithm(samples, 3, 0.5, [2, 0, 0, 4, 5])
model.learn()
model.get_centroids()

print("> sample , allocated cluster : ")
for sample in samples:
    print(sample, model.classify_sample(sample))
    print('\n')

print("> classifying new sample: ")
print(model.classify_sample(np.array([0, -2])))