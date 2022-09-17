from math import tanh, exp
import numpy as np

def heaviside_function_with_threshold(wx, threshold=0, h0=0.5):
    x = round(wx - threshold, 5)
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return 0.0
    elif x == 0.0:
        return h0

def heaviside_function(WX, h0=0.5):
    if WX > 0.0:
        return 1.0
    elif WX < 0.0:
        return 0.0
    elif WX == 0.0:
        return h0

def rectified_linear_unit(x):
    return x if x > 0.0 else 0.0


def leaky_rectified_linear_unit(x, a=0.1):
    return parametric_rectified_linear_unit(x, a)


def parametric_rectified_linear_unit(x, a=0.1):
    return x if x > 0.0 else a * x


def tahn_activation_function(x):
    return tanh(x)


def symmetric_tangent_sigmoid_function(x):
    return 2 / (1 + exp(-2 * x)) - 1


def logarithmic_sigmoid_function(x):
    return round(1 / (1 + exp(-x)),4)

def sgn(x):
    if(x >= 0):
        return 1
    else:
        return -1

def sgn_bagging(x):
    if(x > 0):
        return 1
    elif(x == 0):
        return 1
    else:
        return -1

if __name__=='__main__':
    #print("Heavside Function : ")
    w = [2, 0.5, 1]
    # Augmented notation
    x = [[1, 0, 2],
         [1,2,1],
         [1,-3,1],
         [1,-2,-1],
         [1,0,-1]
         ]
    #for xi in x:
        #print(heaviside_function(np.matmul(w, xi)))
        #print('\n')

    # INPUT -> HIDDEN LAYER.
    patterns = np.array([
        [1,0,1],
        [0,1,1],
        [1,0,0],
        [0,1,0]
    ])

    wji = np.array ([
        [-0.7057, 1.9061, 2.6605, -1.1359],
        [0.4900, 1.9324, -0.4269, -5.1570],
        [0.9438, -5.4160, -0.3431, -0.2931]
    ])

    wjo = np.array([
        [4.8432],
        [0.3973],
        [2.1761]
    ])

    # each column is for Yji
    y = np.vectorize(symmetric_tangent_sigmoid_function)(np.matmul(wji, patterns) + wjo)
    #print(y)
    #print('\n')
    #print('\n')

    # HIDDEN -> OUTPUT LAYER.
    wkj = np.array ([
        [-1.1444, 0.3115, -9.9812],
        [0.0106, 11.5477, 2.6479]
    ])

    wko = np.array ([
        [2.5230],
        [2.6463]
    ])

    #if e has power then add it as decimal points before
    z = np.vectorize(logarithmic_sigmoid_function)(np.matmul(wkj, y) + wko)
    #print(z)

    net_x = 1*2+ 0.5*2+ (-6)*(-3) + (-2)
    y = sgn(logarithmic_sigmoid_function(net_x))
    print(y)



