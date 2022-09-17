import numpy as np

def sum_of_absolute_differences(a, b):
    return np.sum(np.abs(a - b))

def sum_of_squared_differences(a, b):
    return np.sum(np.square(a - b))

def cross_correlation(a, b):
    return np.sum(a * b)

def normalised_cross_correlation(a, b):
    numerator = cross_correlation(a, b)
    denominator = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    return numerator / denominator

def cosine(a, b):
    return normalised_cross_correlation(a, b)

def correlation_coefficient(a, b):
    normalised_a = a - np.mean(a)
    normalised_b = b - np.mean(b)
    numerator = cross_correlation(normalised_a, normalised_b)
    denominator = np.sqrt(np.sum(np.square(normalised_a))) * np.sqrt(np.sum(np.square(normalised_b)))
    return numerator / denominator

def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def ssd(z,t):
    ssdd = 0
    for i in range(0, len(z)):
        ssdd = ssdd + ((z[i] - t[i])**2)
    return ssdd

if __name__ == '__main__':
    T1 = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ])

    I = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ])

    print("Cross Correlation: ")
    print(cross_correlation(T1, I))
    print("Normalised Cross Correlation: ")
    print(round(normalised_cross_correlation(T1, I), 2))
    print("Correlation Coefficient: ")
    print(round(correlation_coefficient(T1, I), 2))
    print("Sum of Absolute Differences: ")
    print(round(sum_of_absolute_differences(T1, I), 2))
    print("Sum of Squared Differences: ")
    print(round(sum_of_squared_differences(T1, I), 2))
    print("Euclidean Distance: ")
    print(round(euclidean_distance(T1, I), 2))



