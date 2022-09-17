from scipy.signal import correlate2d
from numpy import correlate
from math import ceil
import numpy as np

def after_stride(arr, stride):
    y_dim, x_dim = arr.shape
    new_y_dim = ceil(y_dim / stride)
    new_x_dim = ceil(x_dim / stride)
    out = np.zeros((new_y_dim, new_x_dim))
    for y in range(new_y_dim):
        for x in range(new_x_dim):
            out[y, x] = arr[stride * y, stride * x]
    return out


def pad(arr, padding):
    y_dim, x_dim = arr.shape
    out = np.zeros((y_dim + 2 * padding, x_dim + 2 * padding))
    out[padding:y_dim + padding, padding:x_dim + padding] = arr
    return out


def dilate(arr, dilation):
    y_dim, x_dim = arr.shape
    out = np.zeros((dilation * (y_dim - 1) + 1, dilation * (x_dim - 1) + 1))
    for ix, iy in np.ndindex(arr.shape):
        out[dilation * ix, dilation * iy] = arr[ix, iy]
    return out


def one_arr_one_mask(arr, kernel, *, padding=0, stride=1, dilation=1):
    arr = pad(np.array(arr), padding)
    kernel = dilate(kernel, dilation)
    return after_stride(correlate2d(arr, kernel, 'valid'), stride)


def one_one_conv(A, H):
    out = []
    for xi in A:
        row = []
        for i in xi:
            row.append(correlate(i, H, mode='valid'))
        out.append(row)
    print("each box represents one row of the output")
    print(np.array(out))


def conv2D(A, H, stride, dilation, mode, padding=0):
    # Padding

    if (padding != 0):
        A_pad = []
        for x in A:
            A_pad.append(pad(x, padding))
        A = A_pad

    # Dilation
    if (dilation != 0):
        M = []
        for hi in H:
            M.append(dilate(hi, dilation))
        H = M

    if mode == 'multi_channel':
        A_w = []
        for i in range(0, len(H)):
            A_w.append(correlate2d(A[i], H[i], mode='valid'))
        A_out = sum(A_w)
        A_out = after_stride(A_out, stride)
        print(A_out)

    elif mode == 'multi_mask':
        for h in H:
            print(correlate2d(A, h, mode='valid'))


X = np.array([[[0.2,1,0],[-1,0,-0.1], [0.1,0,0.1]],
              [[1,0.5,0.2], [-1,-0.5,-0.2], [0.1,-0.1,0]]])

H = np.array([[[1,-0.1],[1,-0.1]],
              [[0.5,0.5],[-0.5,-0.5]]])
print("> multi channel : ")
conv2D(X,H, stride=1, dilation = 2, mode='multi_channel', padding=0)
print('\n')

X = np.array([[-1,1,0.5,1],[0,-2,-1,0], [0.5,-1,-1,1], [-1,1,0,-2]])

H = np.array([[[2,0.5],[1,-1]],
              [[-1,1],[0,-2]],
              [[2,1],[-1,0.5]]])

print("> multi mask : ")
conv2D(X,H, stride=1, dilation = 0, mode='multi_mask', padding=0)
print('\n')

X = np.array([[[0.2,1,0.5], [1,0.5,-0.5], [0,0.2,-0.1]],
              [[-1,-1,0], [0,-0.5,-0.4], [-0.1,-0.2,0]],
              [[0.1,0.1,0.5],[0,-0.1,0.5], [0.1,0,0.2]]])

H = np.array([1,-1,0.5])
print("> one - one conv : ")
one_one_conv(X,H)
print('\n')
