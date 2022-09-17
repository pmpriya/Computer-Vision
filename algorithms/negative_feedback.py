import numpy as np

def negative_feedback(weights, x , y, alpha = 0.25, epochs=3):
    # This version of the negative feedback algorithm requires augmented notation.

    for epoch in range(1, epochs + 1):
        print("Epoch %i - negative feedback network " % epoch)
        wt = weights.transpose()
        e = x - np.dot(wt,y)
        we = np.dot(weights,e)
        y = y + alpha * we
        wy = np.dot(wt, y)
        print(" > e: %s, we: %s, y:%s, wy: %s"
              % (e.transpose(), we.transpose(), y.transpose(), wy.transpose()))
        print("")

    return y


weights = np.array([
    [1,1,0],
    [1,1,1]
])


epochs = 4

# 3 input neurons
x = np.array([1,1,0])
# 2 output neurons
y = np.array([0,0])
alpha = 0.25

print(negative_feedback(weights, x, y , alpha, epochs=epochs))