from sklearn.ensemble import RandomForestClassifier


def criarModelo(arvores=100):
    return RandomForestClassifier(n_estimators=arvores, random_state=42, n_jobs=-1)


def treinarRandomForest(X_train, y_train, arvores=100):
    model = criarModelo(arvores)
    model.fit(X_train, y_train)

    return model
