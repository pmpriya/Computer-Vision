import numpy as np
from numpy.lib.stride_tricks import as_strided


def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    '''
     2D Pooling

     Parameters:
         A: input 2D array
         kernel_size: int, the size of the window over which we take pool
         stride: int, the stride of the window
         padding: int, implicit zero paddings on both sides of the input
         pool_mode: string, 'max' or 'avg'
     '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride * A.strides[0], stride * A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))


A = np.array([
    [0.2, 1, 0, 0.4],
    [-1, 0, -0.1, -0.1],
    [0.1, 0, -1, -0.5],
    [0.4, -0.7, -0.5, 1]
])

print(pool2d(A, kernel_size=2, stride=2, padding=0, pool_mode='avg'))
print('\n')
print(pool2d(A, kernel_size=2, stride=2, padding=0, pool_mode='max'))
print('\n')
print(pool2d(A, kernel_size=3, stride=1, padding=0, pool_mode='max'))

