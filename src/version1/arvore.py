from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def criarModelo(profundidade=5):
    return DecisionTreeClassifier(max_depth=profundidade, random_state=42)


def treinarArvore(X_train, y_train, profundidade=5):
    model = criarModelo(profundidade)
    model.fit(X_train, y_train)

    return model


def compararProfundidade(X_train, y_train, valores=(3, 5, 7, 10)):
    print("\n------------- Comparando profundidade por validacao cruzada (acuracia) -------------")
    resultados = {}
    for p in valores:
        acuracia = cross_val_score(criarModelo(p), X_train, y_train, cv=cv, scoring="accuracy")
        resultados[p] = acuracia.mean()
        print(f"profundidade = {p:<3} | acuracia media = {acuracia.mean():.4f} | desvio = {acuracia.std():.4f}")

    return max(resultados, key=resultados.get)
