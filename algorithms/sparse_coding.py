import numpy as np

def sparse_coding(x, V, y, _lambda= 1):

    #error = l2 norm ( x - Vt * y )

    r_error = []
    for yi in y:
        val = np.linalg.norm(x.transpose() - np.dot(V.transpose(), yi.transpose()))
        print("L2 norm of reconstruction error: ", val)
        sparsity = np.count_nonzero(yi)
        print("sparsity: ", sparsity)
        r_error.append(val + _lambda * np.count_nonzero(yi))

    print('\n')
    print("reconstruction errors for all y: ")
    print(r_error)
    print(y[np.argmin(r_error)], " for sparse coding")

V = np.array([
    [0.4, -0.6],
    [0.55, -0.45],
    [0.5, -0.5],
    [-0.1, 0.9],
    [-0.5, -0.5],
    [0.9, 0.1],
    [0.5, 0.5],
    [0.45, 0.55]
])

x = np.array([
    [-0.05],
    [-0.95]
])

yt = np.array([
    [1,0,0,0,1,0,0,0],
    [0,0,1,0,0,0,-1,0]
])

sparse_coding(x, V, yt)
print('\n')