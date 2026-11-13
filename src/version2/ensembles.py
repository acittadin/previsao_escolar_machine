from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.pipeline import Pipeline

from avaliacao import compararParametros
from preprocessamento import criarPreparo

# Ensembles combinam vários modelos para errar menos do que um só.
#   Random Forest / Extra Trees: muitas árvores independentes, média das previsões (reduz a variância)
#   Gradient Boosting: árvores em sequência, cada uma corrige o erro das anteriores
#   Voting: junta modelos de famílias diferentes (média das probabilidades)


def criarRandomForest(profundidade=12, folha=30, arvores=200):
    return Pipeline(steps=[
        ("prep", criarPreparo(escalar=False)),
        ("floresta", RandomForestClassifier(n_estimators=arvores, max_depth=profundidade, min_samples_leaf=folha,
                                            class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    ])


def criarExtraTrees(profundidade=14, folha=20, arvores=200):
    # Igual à Random Forest, mas sorteia também os limiares de corte (mais aleatoriedade, menos variância)
    return Pipeline(steps=[
        ("prep", criarPreparo(escalar=False)),
        ("floresta", ExtraTreesClassifier(n_estimators=arvores, max_depth=profundidade, min_samples_leaf=folha,
                                          class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    ])


def criarGradientBoosting(taxa=0.03, profundidade=5, iteracoes=200):
    # early_stopping=False: o número de iterações é fixo (o "separar um pedaço do treino" do early
    # stopping não respeitaria o agrupamento por estudante)
    return Pipeline(steps=[
        ("prep", criarPreparo(escalar=False)),
        ("boosting", HistGradientBoostingClassifier(learning_rate=taxa, max_depth=profundidade, max_iter=iteracoes,
                                                    min_samples_leaf=50, class_weight="balanced",
                                                    early_stopping=False, random_state=42))
    ])


def criarVoting(random_forest, gradient_boosting, regressao):
    # Votação "suave": média das probabilidades dos três modelos (cada um com o seu próprio pré-processamento)
    return VotingClassifier(
        estimators=[("rf", clone(random_forest)), ("gb", clone(gradient_boosting)), ("lr", clone(regressao))],
        voting="soft", n_jobs=1
    )


def compararEnsembles(X_train, y_train, grupos):
    # devolve os melhores parâmetros de cada ensemble
    rf = compararParametros(criarRandomForest, {"profundidade": (6, 12), "folha": (10, 30)}, X_train, y_train, grupos)
    et = compararParametros(criarExtraTrees, {"profundidade": (8, 14), "folha": (5, 20)}, X_train, y_train, grupos)
    gb = compararParametros(criarGradientBoosting, {"taxa": (0.03, 0.1), "profundidade": (3, 5)},
                            X_train, y_train, grupos)
    return rf, et, gb
