import numpy as np
from distance_metrics import euclidean_distance

class fuzzy_kmeans:
    def normalise(self,memberships):
        cumulative = np.sum(memberships, axis=0, keepdims=True)
        return memberships / cumulative

    def get_cluster_centres(self,dataset, memberships, b):
        number_of_clusters, _ = memberships.shape
        centroids = []
        for centroid_i in range(number_of_clusters):
            memberships_exponent = np.power(memberships[centroid_i, :], b)
            numerator = np.sum(memberships_exponent * dataset, axis=1)
            denominator = np.sum(memberships_exponent)
            centroids.append(numerator / denominator)
        return centroids

    def get_memberships(self,dataset, centroids, b):
        _, number_of_samples = dataset.shape
        number_of_clusters = len(centroids)
        memberships = np.zeros((number_of_clusters, number_of_samples))
        for sample_i in range(number_of_samples):
            sample = dataset[:, sample_i]
            for centroid_i, centroid in enumerate(centroids):
                distance = euclidean_distance(sample, centroid)
                memberships[centroid_i, sample_i] = np.power(1 / distance, 2 / (b - 1))
        return self.normalise(memberships)

# passing as S(transpose)
S = np.array([
        [-1, 1, 0, 4, 3, 5],
        [3, 4, 5, -1, 0, 1]
    ])

# passing as Mu (transpose)
mu = np.array([
        [1, 0.5, 0.5, 0.5, 0.5, 0],
        [0, 0.5, 0.5, 0.5, 0.5, 1]
    ])

fkm = fuzzy_kmeans()
iterations = 4
for iteration in range(0, iterations):
    print("###### iteration: {}".format(iteration + 1))
    #print("normalized memberships: ", fkm.normalise(mu))
    cluster_centres = fkm.get_cluster_centres(S, fkm.normalise(mu), 2)
    mu = fkm.get_memberships(S, cluster_centres, 2)
    print(cluster_centres)
    print('\n')
