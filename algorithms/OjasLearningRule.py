import numpy as np


class OjasLearningRule:
    def __init__(self, samples, weights, learning_rate=0.01, epochs=2):
        self.mu = np.mean(np.stack(samples), axis=0)
        self.weights = np.array(weights,  dtype='float64')
        self.load(samples, learning_rate, epochs)
        #self.batch(samples, learning_rate, epochs)

    def load(self, samples, n, epochs):
        zero_mean_samples = [sample - self.mu for sample in samples]

        for epoch in range(1, epochs + 1):
            print("Epoch %i" % epoch)
            #sigma_delta = np.zeros(self.weights.shape, dtype='float64')

            for i, x in enumerate(zero_mean_samples, 1):
                print("Sample %i" % i)
                x_t = np.transpose(x)
                print(" > x_t: ", x_t)
                y = float(np.matmul(self.weights, x))
                print(" > y = wx: ", y)
                sub = x_t - y * self.weights
                print(" > x_t - yw: ", sub)
                delta = n * y * sub
                print(" > ny(x_t - yw): ", delta)
                #sigma_delta += delta

                self.weights += delta
                #print("Sigma delta: ", sigma_delta)
                print("W: ", self.weights.round(2), "\n")

    def project(self, sample):
        return np.matmul(self.weights, sample - self.mu)


    def batch(self, samples, n, epochs):
        zero_mean_samples = [sample - self.mu for sample in samples]

        for epoch in range(1, epochs + 1):
            print("Epoch %i" % epoch)
            sigma_delta = np.zeros(self.weights.shape, dtype='float64')

            for i, x in enumerate(zero_mean_samples, 1):
                print("Sample %i" % i)
                x_t = np.transpose(x)
                print(" > x_t: ", x_t)
                y = float(np.matmul(self.weights, x))
                print(" > y = wx: ", y)
                sub = x_t - y * self.weights
                print(" > x_t - yw: ", sub)
                delta = n * y * sub
                print(" > ny(x_t - yw): ", delta)
                sigma_delta += delta

            self.weights = weights + sigma_delta
                #print("Sigma delta: ", sigma_delta)
            print("W: ", self.weights.round(2), "\n")

#weights = [-1, 0]
#learning_rate = 0.01
### note- use zero mean data
#samples = np.array([
#    [-5,-4],
#    [-2,0],
#    [-0,-1],
#    [0,1],
#    [3,2],
#    [4,2]
#])
#OjasLearningRule(samples, weights, learning_rate)


weights = [-0.2, -0.2, 0.2, -0.0]
learning_rate = 0.01
### note- use zero mean data
x  = np.array([[0.11, 0.11, -0.49, -1.69],
               [1.31, 2.11, 1.41, 0.81],
               [0.61, 0.11, 0.31, -1.69],
               [-1.79,  1.41, -0.89, -2.39],
               [1.31,  0.71, -2.59,  1.21]])

OjasLearningRule(x, weights, learning_rate,epochs=1)

