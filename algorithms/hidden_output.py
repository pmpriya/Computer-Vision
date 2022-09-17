import numpy as np
from activation_functions import symmetric_tangent_sigmoid_function, heaviside_function, logarithmic_sigmoid_function

def compute_hidden_output_mlp(wj, wj0, wk, wk0,x):
    y = []
    z = []

    for i in range(0, len(wj)):
        y_i = logarithmic_sigmoid_function(np.matmul(wj[i], x) + wj0[i])
        y.append(round(y_i, 4))

    print("> hidden units: ")
    print(y)
    # y = [ y1, y2, y3...]

    for i in range(0, len(wk)):
        print("> net j: ", np.matmul(wk[i], y) + wk0[i])
        z_i = np.matmul(wk[i], y) + wk0[i]
        z.append(z_i)

    print("> output units: ")
    print(z)

def compute_hidden(wj, wj0,x):
    y = []

    for i in range(0, len(wj)):
        #y_i = symmetric_tangent_sigmoid_function(np.matmul(wj[i], x) + wj0[i])
        # y = H(WX-theta)
        y_i = heaviside_function(np.matmul(wj[i], x) + wj0[i])
        y.append(np.round_(y_i, 4))

    print("> hidden units: ")
    print(y)
    # y = [ y1, y2, y3...]

if __name__ == '__main__':

    wj = np.array([[0.1, -0.5, 0.4]])
    # w0 = - theta
    w0 = np.array([[0.0]])
    x1 = np.array([0.1,-0.5,0.4])
    x2 = np.array([0.1, 0.5, 0.4])

    #compute_hidden(wj, w0, x1)
    #compute_hidden(wj, w0, x2)

    # w11, w12, w13,...
    wj = np.array([[1, -1],
                   [3, -5],
                   [-1, 3]
                   ])

    wj0 = np.array([[-2],
                    [-3],
                    [3]])

    # w'11, w'12, w'13...
    wk = np.array([[3, 2, 5],
                   [3, 5, -2]
                   ])
    wk0 = np.array([[-3],
                    [-5]
                    ])
    x = np.array([0.4, -0.3])
    compute_hidden_output_mlp(wj, wj0, wk, wk0,x)
