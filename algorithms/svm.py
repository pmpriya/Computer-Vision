import numpy as np
from sklearn import svm


def feature_mapping(samples):
    for i in range(0, len(samples)):
        if(np.linalg.norm(samples[i]) > 2):
            samples[i] = [4 - samples[i][1]/2 + abs(samples[i][0] - samples[i][1]) , 4 - samples[i][0]/2 + abs(samples[i][0] - samples[i][1])]
        else:
            samples[i] = [samples[i][0] - 2, samples[i][1] - 3]

    return samples

def hard_classifier(wt, xi, w0):
    print(x * wt)
    fx = np.dot(x,wt) + w0
    if(fx < 0):
        return -1
    else:
        return 1

def soft_classifier(wt, xi, w0):

    fx = np.dot(x,wt) + w0
    if(fx < -1):
        return -1
    elif(-1 <= fx <= 1):
        return fx
    else:
        return 1

class svm_algorithm():
    def fit(self, x, y ):
        weight = []
        w = []
        r = []
        for i in range(0, len(x)):
            wi = []
            for xi in range(0,len(x)):
                wi.append(np.dot(y[xi],np.dot(x[i], x[xi])))
                #w0
            wi.append(1)
            w.append(wi)

        for yi in y:
            r.append(yi)
        r.append(0)
        w.append(r)

        lambdas = np.dot(np.linalg.inv(w), np.array(r).transpose())
        print("> values of lambdas and w0: ")
        print(lambdas)

        for i in range(0, len(x)):
            delta = lambdas[i]*x[i]*y[i]
            weight.append(list(delta.round(3)))
        print(" Weights: ")
        weight = np.array(weight).sum(axis=0)
        print(weight)

        return weight, lambdas

# support vectors
#samples = [
#    [-2.5,-1.5],
#    [-1,0],
#    [-2.5,1.5],
#    [0.5,-0.5],
#    [2.5,-1.5],
#    [1,0],
#    [2.5,1.5],
#    [-0.5,-1]
#]

samples = [
    [3,3],
    [9,7],
    [5,5],
    [7,9],
    [-1,-2],
    [-1,-4],
    [-3,-4],
    [-3,-2]
    ]

y = []
#labels = [
#    -1,-1,-1,-1,1,1,1,1
#]

labels = [
    1,1,1,1,-1,-1,-1,-1
]

#samples.... = feature_mapping(samples)
#print(samples)

clf = svm.SVC(kernel='linear')
clf.fit(samples, labels)
x = clf.support_vectors_
print("> support vectors: ", x)
for i in clf.support_:
    y.append(labels[i])


#x = np.array([[-2 , 4 ],
#              [ -1 , 1],
#              [ 2, 4 ]
#            ])
#y = [1,-1,-1]
wt, lambdas = svm_algorithm().fit(x,y)