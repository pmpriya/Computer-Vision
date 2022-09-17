import numpy as np


def linear_discriminant_function(a,y):
    gx = np.matmul(a,y)
    if gx > 0:
        return 1
    else:
        return -1

def linear_discriminant_function2(w,x, w0):
    gx = np.matmul(np.transpose(w), x) + w0
    if gx > 0:
        return 1
    else:
        return 2



if __name__ == '__main__':
    #aty => use augmented notation

    #a1 = np.array([1,0.5,0.5])
    #a2 = np.array([-1, 2, 2])
    #a3 = np.array([2, -1, -1])

    at = np.array([1.6,9.4,8.2,1.8,5.8,0.7])
    y = np.array([[1, -2.0,1.5],
                 [1, 2.0,1.5],
                 [1, -1.0,0.0],
                 [1, 0.5,-0.5]])
    print(np.transpose(y))

    linear_discriminant_function(at,y)



