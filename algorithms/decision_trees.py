from math import log2


def metrics(lst):
    bucket_counts = {}
    number_of_elements = len(lst)

    for element in lst:
        if element not in bucket_counts:
            bucket_counts[element] = 0
        bucket_counts[element] += 1

    entropy = 0
    gini_impurity = 0
    for count in bucket_counts.values():
        pi = count / number_of_elements
        entropy += -1 * pi * log2(pi)
        gini_impurity += pi * (1 - pi)

    print("Entropy: %f" % entropy)
    print("GINI impurity: %f" % gini_impurity)


if __name__ == "__main__":
    metrics([1, -1, +1, +1, -1, -1, -1, +1])
    metrics([1, 1, -1])