import numpy as np


class HebbianLearningRule:
    def __init__(self, samples, weights, learning_rate=0.01, epochs=2):
        self.mu = np.mean(np.stack(samples), axis=0)
        self.weights = np.array(weights,  dtype='float64')
        self.load(samples, learning_rate, epochs)

    def load(self, samples, n, epochs):
        zero_mean_samples = [sample - self.mu for sample in samples]

        for epoch in range(1, epochs + 1):
            print("Epoch %i" % epoch)

            for i, x in enumerate(zero_mean_samples, 1):
                print("Sample %i" % i)
                x_t = np.transpose(x)
                print(" > x_t: ", x_t)
                y = float(np.matmul(self.weights, x))
                print(" > y = wx: ", y)
                delta = n * y * x_t
                print(" > nyx_t ", delta)
                self.weights += delta
                print("W: ", self.weights.round(2), "\n")

    def project(self, sample):
        return np.matmul(self.weights, sample - self.mu)

weights = [0.5, -0.2]
learning_rate = 0.1
samples = np.array([
    [0,1],
    [1,2],
    [3,1],
    [-1,-2],
    [-3,-2]
])

HebbianLearningRule(samples, weights, learning_rate)