import numpy as np
from math import sqrt

class BatchNormalisation:
    def __init__(self, *args, beta = 0.1, gamma = 0.4, epsilon=0.2):
        len = args[0]
        big = []
        bn_list = []

        for i in range(1,len+1):
            big.append(args[i])

        for i in range(0, np.asarray(big[0]).shape[0]):
            for j in range(0, np.asarray(big[0]).shape[0]):
                store = []
                dummy = []
                for x in big:
                    store.append(x[i][j])
                e = np.mean(np.array(store))
                v = np.var(np.array(store))
                for x in big:
                    bn = beta + gamma * (( x[i][j] - e) / sqrt(v + epsilon))
                    dummy.append(round(bn,4))
                bn_list.append(dummy)

        print(" ### each row represents batch normalisation for one sample ### ")
        #print(np.asarray(bn_list))
        print(np.asarray(bn_list).transpose())

x1 = [[0.9,-0.6, 0.0], [0.3, 0.9, 0.1], [-0.7, 0.0, 0.8]]
x2 = [[0.0, 0.0, -0.5], [-0.9, -0.7, -0.4], [0.6, -0.4, 0.9]]
x3 = [[-0.2, -0.4, 0.1], [-0.2, 0.8, -0.9], [-0.1, 0.1, -0.3]]


BatchNormalisation(3, x1,x2,x3)