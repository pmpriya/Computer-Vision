import numpy as np

class BatchCompetitiveLearningAlgorithm:
    def __init__(self, samples, centroids, n, selection_order):
        self.samples = [np.array(sample) for sample in samples]
        self.centroids = [np.array(centroid) for centroid in centroids]
        self.selection_order = selection_order
        self.n = n

    def index_of_nearest_cluster(self, sample):
        number_of_clusters = len(self.centroids)
        return max(range(number_of_clusters), key=lambda i: np.dot(self.centroids[i], sample))

    def learn(self):
        for index in self.selection_order:
            sample = self.samples[index]
            j = self.index_of_nearest_cluster(sample)
            self.centroids[j] = self.centroids[j] + self.n * np.dot(sample,self.centroids[j])
            self.centroids[j] = self.centroids[j]/ np.linalg.norm(self.centroids[j])
            print(" weight updated: ")
            print(list(self.centroids))

    def get_centroids(self):
        return self.centroids

    def classify_sample(self, sample):
        centroid_index = self.index_of_nearest_cluster(sample)
        return self.centroids[centroid_index]


samples = [(1,-1, 3), (1,1, 4), (1,0, 5), (1,4, -1), (1,3, 0), (1,5,1)]
norm_samples = samples / np.linalg.norm(samples)
centroids = [(1,-0.5, 1.5), (1,0, 2.5), (1,1.5, 0)]
cla = BatchCompetitiveLearningAlgorithm(norm_samples, centroids, 0.5, [0, 1, 2, 3, 4,5])
cla.learn()
cla.get_centroids()
print('\n')
print("> sample , allocated cluster : ")
for sample in samples:
    print(sample, cla.classify_sample(sample))
#print(cla.classify_sample(np.array([0, -2])))


