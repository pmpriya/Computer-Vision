import numpy as np
from math import log, exp

class Bagging:
    def __init__(self, classifiers, samples, labels):
        self.classifiers = classifiers
        self.h = []
        self.a = []
        self.learn(samples, labels)
        self.improve(samples, labels)

    @staticmethod
    def sign(x):
        return +1 if x >= 0 else -1

    def learn(self, samples, labels):
        f = []
        for sample_i, (sample, label) in enumerate(zip(samples, labels)):
            counter = 0
            for classifier_i, classifier in enumerate(self.classifiers):
                counter += classifier(sample)
            f.append(self.sign(counter/8))
        print("> fi(x) = << for each label >> ", f)
        counter = 0
        for x in range(0, len(f)):
            if(f[x] != labels[x]):
                counter += 1
        print("> f(x) = ", round(counter/len(labels),2))

    def improve(self, samples, labels):
        print('\n')
        print(" Improving Performance of Bagging Classifier: ")
        training_error = []
        for classifier_i, classifier in enumerate(self.classifiers):
            misclassified = 0
            for sample_i, (sample, label) in enumerate(zip(samples, labels)):
                if(classifier(sample) != label):
                    misclassified += 1
            training_error.append(round(misclassified/len(labels),2))
            #print("> Training error of classifier {} : {} ", classifier_i, round(misclassified/len(labels),2))

        print(training_error)
        improved_classifiers = []

        print("> weak classifiers: ", end="")
        for error_i in range(0, len(training_error)):
            if(training_error[error_i] <= 0.50):
                improved_classifiers.append(error_i)
                print(error_i+1, end=" ")

        f = []
        for sample_i, (sample, label) in enumerate(zip(samples, labels)):
            counter = 0
            for classifier_i in improved_classifiers:
                counter += self.classifiers[classifier_i](sample)
            f.append(self.sign(counter / 8))
        print('\n')
        print("> fi(x) = << for each label >> ", f)
        counter = 0
        for x in range(0, len(f)):
            if (f[x] != labels[x]):
                counter += 1
        print("> f(x) = ", round(counter / len(labels), 2))






samples = [
    np.array([1 , 0]),
    np.array([-1 , 0]),
    np.array([0 , 1]),
    np.array([0 , -1])
]

labels = [1,1,-1,-1]

classifiers = [
    lambda x: 1 if x[0] > -0.5 else -1,
    lambda x: -1 if x[0] > -0.5 else 1,
    lambda x: 1 if x[0] > 0.5 else -1,
    lambda x: -1 if x[0] > 0.5 else 1,
    lambda x: 1 if x[1] > -0.5 else -1,
    lambda x: -1 if x[1] > -0.5 else 1,
    lambda x: 1 if x[1] > 0.5 else -1,
    lambda x: -1 if x[1] > 0.5 else 1
]

Bagging(classifiers, samples, labels)