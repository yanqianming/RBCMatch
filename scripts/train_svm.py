import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

data = np.load("../data/20240511/rbc_ftrs.npy")
X = data[:, :-1]
Y = data[:, -1]

KF = StratifiedKFold(5)
for train_index, test_index in KF.split(X, Y):
    X_train, X_test = X[train_index, :], X[test_index, :]
    Y_train, Y_test = Y[train_index], Y[test_index]
    predictor = SVC(kernel="poly", degree=3)
    predictor.fit(X_train, Y_train)

    result = predictor.predict(X_test)
    print(f"F1-score: {f1_score(Y_test, result, average='weighted')}")
    print(f"Precision: {precision_score(Y_test, result, average='weighted')}")
    print(f"Accuracy: {recall_score(Y_test, result, average='weighted')}")
