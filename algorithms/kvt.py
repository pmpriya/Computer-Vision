# Implementation of Karhunen-Loève Transform
# Method for implementing PCA
# Question 2 Tutorial 10
import numpy as np



class kvt:

    def fit(self, x, dimension):

        m = x.mean(axis=0)
        print("m: ")
        print(np.transpose(m))
        # making it zero-mean
        x_m = (x - m)
        print("Xm: ")
        print(np.transpose(x_m))
        # calc covariance matrix of zero mean

        c = np.cov(np.transpose(x_m), bias=True)
        print("C:")
        print(c)
        # Finding eigen vectors of covariance matrix
        d,v = np.linalg.eigh(c)
        print("V:")
        print(v)
        print("D:")
        print(np.diag(d))

        dimension = -(dimension)
        ind = d.argsort()[dimension:][::-1]

        v_hat = v[:,ind]
        print("v_hat")
        print(v_hat)
        v_hatT = np.transpose(v_hat)
        x_kl = np.matmul(v_hatT, np.transpose(x_m))
        print("> output::: ")
        print(np.transpose(x_kl))
        #return np.transpose(x_kl)
        return v_hatT

    def new_sample(self,v_hatT, new_sample):
        new_targets = np.matmul(v_hatT, np.transpose(new_sample))
        print("new target (each col is a converted sample)")
        print(new_targets)

def propotion_variance(x_kl):
    m = np.mean(x_kl)
    x_m = (x_kl - m)
    c = np.round_(np.cov(np.transpose(x_m), bias=True),4)
    print("> covariance of output y: ")
    print(c)

    sum = np.sum(c)
    print("> total of eigenvalues: ")
    print(sum)
    print("> propotion variance for 1st PC: ", np.sum(c[:,0])/sum)
    print("> propotion variance for 2nd PC: ", np.sum(c[:,1])/sum)
    print("> propotion variance by both components: ", (np.sum(c[:,0]) + np.sum(c[:,1])) /sum)

x  = np.transpose(np.array([[4,0,2,-2],
               [2,-2,4,0],
               [2,2,2,2]
               ]))

samples = np.array([
    [-4.6,-3],
    [-2.6,0],
    [-0.6,-1],
    [2.4,2],
    [5.4,2]
])

clf = kvt()
#x_kl = clf.fit(x,2)
print("> 2d: ")
v_hatT = clf.fit(x,2)
new_sample = np.transpose(np.array([3,-2,5]))
clf.new_sample(v_hatT, new_sample)
print('\n')
print("> 1d: ")
v_hatT = clf.fit(x,1)
clf.new_sample(v_hatT, new_sample)

#propotion_variance(np.transpose(x_kl))