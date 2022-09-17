import numpy as np


def sequential_widrow_hoff(Y, b, weights, learning_rate=0.1, epochs=3):
    print(Y)
    print(b)
    print(weights)

    for epoch in range(1, epochs + 1):
        print("Epoch %i - Sequential Widrow Hoff" % epoch)
        for sample, margin in zip(Y, b):
            prediction = np.matmul(np.transpose(sample), weights)
            weights = weights + learning_rate * (margin - prediction) * sample
            print(" > Sample: %s, g(x): %.1f, weights: %s"
                  % (str(sample.transpose()), float(prediction), str(weights.transpose())))
        print("")
    return weights


# Augmented and normalised samples
Y = np.array([
    [1,0,2],
    [1,1,2],
    [1,2,1],
    [-1,3,-1],
    [-1,2,1],
    [-1,3,2]
    ])

weights = [1,0,0]

b = [1,1,1,1,1,1]
print(sequential_widrow_hoff(Y,b,weights, epochs=2))


