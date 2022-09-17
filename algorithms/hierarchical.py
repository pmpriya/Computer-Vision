import math
import numpy as np

class HierarchicalClustering():


    def __init__(self, n_classes, dataset, similarity_method="single-link"):
        self.n = n_classes
        # Dataset of type np.array([[x11, x12], [x12, x22],...,[x1n, x2n]])
        self.dataset = dataset
        # either "single-link", "complete-link", "group-average", or "centroid"
        self.similarity_method = similarity_method
        # The initial clusters of type np.array([[x11, x12], [x12, x22],...,[x1n, x2n]])
        self.clusters = [np.array([s]) for s in
                         self.dataset]

    def run(self):

        iteration = 1
        while (len(self.clusters) != self.n):
            closest_clusters = None
            closest_distance = None
            # Find the two closest clusters
            for i, cluster in enumerate(self.clusters):
                closest_cluster_to_i, distance_to_i = self.find_closest_cluster(i)  # TODO
                if not closest_distance or distance_to_i < closest_distance:
                    closest_clusters = [i, closest_cluster_to_i]
                    closest_distance = distance_to_i

            # Merge closest_clusters
            merged_clusters = [self.clusters[closest_clusters[0]], self.clusters[closest_clusters[1]]]
            self.merge_clusters(closest_clusters[0], closest_clusters[1])
            self.print_iteration(iteration, merged_clusters, closest_distance)
            iteration += 1
        self.print_final()

    def find_closest_cluster(self, cluster_index):

        closest_cluster_index = None
        closest_distance = None
        for i, cluster in enumerate(self.clusters):
            similarity_method_options = {
                "single-link": lambda: self.get_single_link_distance(self.clusters[cluster_index], self.clusters[i]),
                "complete-link": lambda: self.get_complete_link_distance(self.clusters[cluster_index],
                                                                         self.clusters[i]),
                "group-average": lambda: self.get_average_link_distance(self.clusters[cluster_index], self.clusters[i]),
                "centroid": lambda: self.get_centroid_distance(self.clusters[cluster_index], self.clusters[i])}
            if cluster_index != i:
                func = similarity_method_options.get(self.similarity_method, lambda: "Invalid")
                distance = func()
                if not closest_distance or distance < closest_distance:
                    closest_cluster_index = i
                    closest_distance = distance

        return closest_cluster_index, closest_distance

    def get_single_link_distance(self, cluster_a, cluster_b):
        minimum_distance = None
        for a in cluster_a:
            for b in cluster_b:
                dist = np.linalg.norm(a - b)
                if not minimum_distance or minimum_distance > dist:
                    minimum_distance = dist
        return minimum_distance

    def get_complete_link_distance(self, cluster_a, cluster_b):
        maximum_distance = None
        for a in cluster_a:
            for b in cluster_b:
                dist = np.linalg.norm(a - b)
                if not maximum_distance or maximum_distance < dist:
                    maximum_distance = dist
        return maximum_distance

    def get_average_link_distance(self, cluster_a, cluster_b):
        distances = np.array([])
        for a in cluster_a:
            for b in cluster_b:
                dist = np.linalg.norm(a - b)
                distances = np.append(distances, np.array([dist]))
        return np.average(distances)

    def get_centroid_distance(self, cluster_a, cluster_b):
        centroid_a = np.average(cluster_a)
        centroid_b = np.average(cluster_b)
        dist = np.linalg.norm(centroid_a - centroid_b)
        return dist

    def merge_clusters(self, cluster_index_a, cluster_index_b):
        merged_cluster = np.concatenate((self.clusters[cluster_index_a], self.clusters[cluster_index_b]))
        self.clusters = [self.clusters[i] for i in range(0, len(self.clusters)) if
                         i != cluster_index_a and i != cluster_index_b]
        self.clusters.append(merged_cluster)

    def print_iteration(self, i, merged_clusters, distance):
        print(f"End of iteration {str(i)} ")
        print(
            f"Merged clusters {merged_clusters[0]} and {merged_clusters[1]}. Distance between the clusters was {distance}.")
        print(f"There are {str(len(self.clusters))} clusters:")
        for j, c in enumerate(self.clusters):
            print(f"Cluster {j} -> {self.clusters[j]}")
        print("\n")

    def print_final(self):
        print("FINAL CLUSTERS:")
        for j, c in enumerate(self.clusters):
            print(f"Cluster {j} -> {self.clusters[j]}")


if __name__ == '__main__':
    """ parameters for hierarchical clustering
           c -> the number of classes to cluster.
           x -> The dataset to cluster.
           similarity_method -> the distancing method to use.
       """
    # Dataset of type np.array([[x11, x12], [x12, x22],...,[x1n, x2n]])
    c = 3
    x = [[-1, 3],
         [1, 2],
         [0, 1],
         [4, 0],
         [5, 4],
         [3, 2]]

    cluster = HierarchicalClustering(c, x, "single-link")
    cluster.run()