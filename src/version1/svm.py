from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def criarModelo(c=1):
    # SVM usa distância: precisa padronizar as colunas
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=c, random_state=42))
    ])


def treinarSVM(X_train, y_train, c=1):
    model = criarModelo(c)
    model.fit(X_train, y_train)

    return model
