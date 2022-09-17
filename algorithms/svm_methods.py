from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

X = np.array([
     [10, 10],
     [8, 10],
     [-5, 5.5],
     [-5.4, 5.5],
     [-20, -20],
     [-15, -20]
])
y = np.array([0, 0, 1, 1, 2, 2])

clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y)
print(clf.support_vectors_)
#print('\n')
print(clf.predict(X))
#print("accuracy: ", clf.score(X, y))

clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
#print("support vectors: ",clf.support_vectors_)
#print('\n')
print(clf.predict(X))
#print("accuracy: ", clf.score(X, y))
