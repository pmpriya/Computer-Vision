from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def errorRate(y_true, y_pred):
    return 1- accuracy(y_true,y_pred)

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def f1score(y_true, y_pred):
    return f1_score(y_true, y_pred)

def confusionMatrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return [tn, fp, fn, tp]




y_true = [1,1,0,1,0,1,1]
y_pred = [1,0,1,1,0,1,0]

print("confusionMatrix :")
print(confusionMatrix(y_true,y_pred))

print("accuracy :")
print(accuracy(y_true,y_pred))

print("error rate :")
print(errorRate(y_true,y_pred))

print("recall :")
print(recall(y_true,y_pred))

print("precision :")
print(precision(y_true,y_pred))

print("f1 score :")
print(f1score(y_true,y_pred))