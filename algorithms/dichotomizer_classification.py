import numpy as np


# Based on Question 1 in Tutorial 2:

def makeClassification(weight, intercept, feature_vector):
    weight = np.array(weight)
    feature_vector = np.array(feature_vector)

    wx = np.dot(weight, feature_vector)

    # Finding g(x):
    gx = wx + intercept

    if gx > 0:
        print("Feature Vector {} is in Class 1.".format(feature_vector))
    else:
        print("Feature Vector {} is in Class 2.".format(feature_vector))

 # PASS ONLY AUGMENTED VECTOR
def makeClassificatio_with_a(weight, feature_vector):
    weight = np.array(weight)
    feature_vector = np.array(feature_vector)

    gx = np.dot(weight, feature_vector)

    if gx > 0:
        print("Feature Vector {} is in Class 1.".format(feature_vector))
    else:
        print("Feature Vector {} is in Class 2.".format(feature_vector))



def discriminant_function1(x):
    gx = (x[0]*x[0]) - (x[2]*x[2]) + 2*x[1]*x[2] + 4*x[0]*x[1] + 3*x[0] - 2*x[1] + 2
    if gx > 0:
        return 1
    else:
        return 2


def discriminant_function2(x, A, b, c):
    gx = np.matmul(np.matmul(np.transpose(x), A),x) + np.matmul(np.transpose(x), b)+ c
    if(gx > 0):
        return 1
    else:
        return 2


if __name__ == '__main__':
    # determine the class of the following feature vectors:
    # Change these parameters to fit the question at hand.

    #x1 = [1,1,1,1,1]
    #x2 = [1,0,-1,0,1]
    #a = [-2,1,2,1,2]

    w = [-1, -2]
    w0 = 3
    x1 = [2.5, 1.5]
    makeClassification(w,w0, x1)
    #makeClassificatio_with_a(a, x1)

    #x2 = np.array([1, 1])
    #A = np.array([[2,1],
    #              [1,4]])
    #b = np.array([1,2])
    #c = -3
    #print(discriminant_function2(x1,A,b,c))
    #print(discriminant_function2(x2,A,b,c))




