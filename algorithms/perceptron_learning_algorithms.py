import numpy as np

def batch_perceptron_learning_algorithm(weights, samples, learning_rate=1, epochs=3):
    # This version of the batch perceptron learning algorithm requires normalisation and augmented notation.
    weights = np.array(weights)
    samples = [np.array(sample) for sample in samples]

    for epoch in range(1, epochs + 1):
        print("Epoch %i - batch perceptron learning algorithm" % epoch)
        sigma_y = np.zeros(weights.shape)
        for sample in samples:
            prediction = float(np.matmul(weights.transpose(), sample))
            print(" > Sample: %s, g(x): %.1f" % (str(sample.transpose()), prediction))
            if prediction <= 0:
                sigma_y += sample
        weights = weights + learning_rate * sigma_y
        print(" > Sigma y: %s, new weights: %s\n" % (str(sigma_y.transpose()), str(weights.transpose())))

    return weights


def sequential_perceptron_learning_algorithm(weights, samples, learning_rate=1, epochs=3):
    # This version of the batch perceptron learning algorithm requires normalisation and augmented notation.
    weights = np.array(weights)
    samples = [np.array(sample) for sample in samples]

    for epoch in range(1, epochs + 1):
        print("Epoch %i - sequential perceptron learning algorithm" % epoch)
        for sample in samples:
            prediction = float(np.matmul(weights.transpose(), sample))
            if prediction <= 0:
                weights = weights + learning_rate * sample
            print(" > yt: %s, g(x): %.1f, wk*yt: %s"
                  % (str(sample.transpose()), prediction, str(weights.transpose())))
        print("")

    return weights


def sequential_multi_class_perceptron_learning_algorithm(weights, samples, learning_rate=1, epochs=3, classes=3):
    # This version of the batch perceptron learning algorithm requires normalisation and augmented notation.
    w = np.array(weights)
    y = [np.array(sample) for sample in samples]

    a = [[0] * len(samples[0]) for i in range(0, classes)]
    g = [0 for i in range(0, classes)]

    for epoch in range(1, epochs + 1):
        print("Epoch %i - sequential multiclass perceptron learning algorithm" % epoch)
        for k in range(0, len(y)):
            for i in range(0, len(a)):
                g[i] = np.matmul(np.transpose(a[i]), y[k])
            j = 0
            max_g = max(g)
            for i, gi in enumerate(g):
                if (gi == max_g):
                    j = i

            print(" > at: %s, yt: %s, g: %s, j: %s, w: %s"
                  % (str(np.array(a)), str(np.array(y[k])), str(np.array(g)), str(j + 1), str(w[k])))

            if ((j + 1) != w[k]):
                # update weights
                a[w[k] - 1] = a[w[k] - 1] + learning_rate * y[k]
                a[j] = a[j] - learning_rate * y[k]

        print("")

def augment(x):
    return np.vstack((np.array([[1]]), x))

def linear_discriminant(x,a):
    # g(x) = at * y
    # a = np.array([[-5], [2], [1]])
    g = lambda x: np.matmul(a.transpose(), augment(x))
    return g(x)

def min_sqaured_error(y, b):
    # Ya = b
    # finding a
    y_pseudo_inverse = np.linalg.pinv(y)
    return np.matmul(y_pseudo_inverse, b)


#print("linear discriminant function: ")
a = np.array([[-5], [2], [1]])
x = np.array([[2], [2]])
#print(linear_discriminant(x,a))
#print('\n')

#print("Batch Perceptron Learning Algorithm: ")

weights = [-25,6,3]
# Augmented and Sample Normalised
samples = np.array([
    [1,1,5],
    [1,2,5],
    [-1,-4,-1],
    [-1,-5,-1]
    ])
#print(batch_perceptron_learning_algorithm(weights, samples))
#print('\n')

print("Sequential Perceptron Learning Algorithm: ")
weights = [-25,5,2]
# Augmented and Sample Normalised
samples = np.array([
    [1,5,1],
    [1,5,-1],
    [1,7,0],
    [-1,-3,0],
    [-1,-2,-1],
    [-1,-1,1]
    ])

print(sequential_perceptron_learning_algorithm(weights, samples, epochs=2))
print('\n')

#print("Min Squared Error: ")
y = np.array([
        [1, 0, 2],
        [1, 1, 2],
        [1, 2, 1],
        [-1, 3, -1],
        [-1, 2, 1],
        [-1, 3, 2] ])
b = np.array([[1], [1], [1], [1], [1], [1]])
#print(min_sqaured_error(y,b))
#print('\n')

weights = [1,1,2,2,3]
# Augmented
samples = np.array([
    [1,1,1],
    [1,2,0],
    [1,0,2],
    [1,-1,1],
    [1,-1,-1]
    ])

#sequential_multi_class_perceptron_learning_algorithm(weights, samples, learning_rate=1, epochs=3, classes=3)


