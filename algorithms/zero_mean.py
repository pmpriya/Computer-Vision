import numpy as np

x  = np.array([[1,2],
               [3,5],
               [5,4],
               [8,7],
               [11,7]])

m = x.mean(axis=0)
print("m: ")
print(np.transpose(m))
        # making it zero-mean
x_m = (x - m)
print("Xm: ")
print(np.transpose(x_m))