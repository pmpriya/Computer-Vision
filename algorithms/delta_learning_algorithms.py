import numpy as np
from activation_functions import heaviside_function


def sequential_delta_learning_rule(weights, samples, labels, learning_rate=1, epochs=3):
    # This version of the batch perceptron learning algorithm requires augmented notation.

    for epoch in range(1, epochs + 1):
        print("Epoch %i - sequential delta learning rule" % epoch)
        for sample, label in zip(samples, labels):
            prediction = heaviside_function(np.matmul(weights, sample))
            ty = label - prediction
            delta = learning_rate * (ty) * sample.transpose()
            #weights = weights + learning_rate * (ty) * sample.transpose()
            weights = weights + delta
            print(" > Sample: %s, y: %.1f, ty:%s, delta: %s, weights: %s"
                  % (str(sample.transpose()), prediction, ty, delta, str(weights)))
        print("")

    return weights


def batch_delta_learning_rule(weights, samples, labels, learning_rate=1, epochs=3):
    # This version of the batch perceptron learning algorithm requires augmented notation.

    for epoch in range(1, epochs + 1):
        print("Epoch %i - sequential delta learning rule" % epoch)
        delta = np.zeros(weights.shape)

        for sample, label in zip(samples, labels):
            prediction = heaviside_function(np.matmul(weights, sample))
            ty = label - prediction
            delta_copy = (ty) * sample.transpose()
            delta += (ty) * sample.transpose()
            print(" > Sample: %s, y: %.1f, ty: %s, delta: %s" % (str(sample.transpose()), prediction, ty,delta_copy))

        weights = weights + delta
        print(" > New weights: %s" % weights)
        print("")

    return weights


# note - theta has to be negative
weights = np.array([0.5, 1, 1])
# only augmented notation
#samples = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
#labels = np.array([0,0,0,1])
samples = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
labels = np.array([0,0,0,1])
print("Sequential Learning Rule : ")
print(sequential_delta_learning_rule(weights, samples, labels, epochs=7))
print('\n')
print('\n')

#print(" Batch Learning Rule : ")
#print(batch_delta_learning_rule(weights, samples, labels, epochs=7))

