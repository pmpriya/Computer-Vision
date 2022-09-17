import numpy as np
# Extreme Learning Machine

def heaviside_function(x, threshold=0, h0=0.5):
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return 0.0
    elif x == 0.0:
        return h0

# Extreme Learning Machine

class RandomProjections():
    def __init__(self, v, weights, samples):
        # hidden layer
        vx = np.matmul(v,samples)
        y = []
        # Augment first, same size as that of the samples
        y.append([1,1,1,1])
        for x in vx:
            xy = []
            for val in x:
                xy.append(heaviside_function(val))
            y.append(xy)

        y  = np.array(y)
        print("> response of the hidden neuron: ")
        print(y)
        # output layer
        z = np.matmul(weights,y)
        print(" response of the output neuron: ")
        print(z)

    def find_w(self, v, samples, z):
        # hidden layer
        vx = np.matmul(v,samples)
        y = []
        # Augment first, same size as that of the samples
        y.append([1,1,1,1])
        for x in vx:
            xy = []
            for val in x:
                xy.append(heaviside_function(val))
            y.append(xy)

        y  = np.array(y)
        print(y)
        # w = z y^-1
        return np.matmul(z,np. linalg. inv(y))


v = np.array([
    [-0.62, 0.44, -0.91],
    [-0.81, -0.09, 0.02],
    [0.74, -0.91, -0.60],
    [-0.82, -0.92, 0.71],
    [-0.26, 0.68, 0.15],
    [0.80, -0.94, -0.83]
])

weights = np.array([
    0,0,0,-1,0,0,2
])

# Using augmented notation
#format : [1,1,1,1], [x1[0], x2[0], x3[0], x4[0]], [x1[1], x2[1]...]
samples = np.array([
    [1,1,1,1],
    [0,0,1,1],
    [0,1,0,1]
])

RandomProjections(v, weights, samples)