from sklearn.metrics import accuracy_score

def misclassification_rate(y_true, y_pred):
    return 1 - accuracy_score(y_true, y_pred)
