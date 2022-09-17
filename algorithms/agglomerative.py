import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, labels_):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels = labels_)

x = [[-1,3],
             [1,2],
             [0,1],
             [4,0],
             [5,4],
             [3,2]]
model = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='single')
print(" 5 clusters ")
model = model.fit(x)
print(model.labels_)

model = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='single')
print(" 4 clusters ")
model = model.fit(x)
print(model.labels_)

model = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')
print(" 3 clusters ")
model = model.fit(x)
print(model.labels_)

#plt.title('Hierarchical Clustering Dendrogram')

#plot_dendrogram(model, model.labels_)
#plt.show()