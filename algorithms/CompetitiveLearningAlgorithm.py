import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


class CompetitiveLearningAlgorithm:
    def __init__(self, samples, centroids, n, selection_order):
        self.samples = [np.array(sample) for sample in samples]
        self.centroids = [np.array(centroid) for centroid in centroids]
        self.selection_order = selection_order
        self.n = n

    def index_of_nearest_cluster(self, sample):
        number_of_clusters = len(self.centroids)
        return min(range(number_of_clusters), key=lambda i: euclidean_distance(self.centroids[i], sample))

    def learn(self):

        for index in self.selection_order:
            sample = self.samples[index]
            j = self.index_of_nearest_cluster(sample)
            self.centroids[j] = self.centroids[j] + self.n * (sample - self.centroids[j])


    def get_centroids(self):
        return self.centroids

    def classify_sample(self, sample):
        centroid_index = self.index_of_nearest_cluster(sample)
        return self.centroids[centroid_index]


samples = [(-1, 3), (1, 4), (0, 5), (4, -1), (3, 0), (5,1)]

centroids = [(-0.5, 1.5), (0, 2.5), (1.5, 0)]
cla = CompetitiveLearningAlgorithm(samples, centroids, 0.1, [2, 0, 0, 4, 5])
cla.learn()

m1, m2, m3 = cla.get_centroids()
print(m1)
print(m2)
print(m3)

print("> sample , allocated cluster : ")
for sample in samples:
    print(sample, cla.classify_sample(sample))
    print('\n')

print(cla.classify_sample(np.array([0, -2])))


