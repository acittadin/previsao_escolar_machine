from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score, train_test_split

# 3 rodadas, agrupando por estudante (o mesmo aluno pode aparecer em mais de uma oferta)
cv = GroupKFold(n_splits=3)


def pontuacao(modelo, X):
    # Escore contínuo de risco: probabilidade da classe 1 ou, no SVM, a distância até a margem.
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def metodoPontuacao(modelo):
    return "predict_proba" if hasattr(modelo, "predict_proba") else "decision_function"


# ----------------------------------------------------------------------------------------------------
# VALIDAÇÃO (usa somente o treino, o teste não é tocado)
# ----------------------------------------------------------------------------------------------------

def compararParametros(criar, grade, X_train, y_train, grupos, tamanho_amostra=None):
    # Testa todas as combinações da grade por validação cruzada e devolve a melhor.
    # A métrica é a PR-AUC (average precision): serve bem para classes raras como a evasão.
    # tamanho_amostra: o SVM RBF é lento, então a busca pode usar uma AMOSTRA do treino.
    if tamanho_amostra is not None and len(X_train) > tamanho_amostra:
        indices = np.random.RandomState(42).choice(len(X_train), tamanho_amostra, replace=False)
        X_train, y_train, grupos = X_train.iloc[indices], y_train.iloc[indices], grupos[indices]

    nomes = list(grade.keys())
    print(f"\n------------- Comparando {', '.join(nomes)} por validacao cruzada ({len(X_train)} linhas) -------------")
    resultados = {}
    for valores in product(*grade.values()):
        parametros = dict(zip(nomes, valores))
        pr = cross_val_score(criar(**parametros), X_train, y_train, groups=grupos, cv=cv,
                             scoring="average_precision")
        resultados[valores] = pr.mean()
        print(f"{str(parametros):<55} | PR-AUC media = {pr.mean():.4f} | desvio = {pr.std():.4f}")

    melhor = max(resultados, key=resultados.get)
    melhores = dict(zip(nomes, melhor))
    print(f"melhor na validacao e {melhores}")

    return melhores


def escolherLimiar(modelo, X_train, y_train, grupos):
    # O limiar de decisão é escolhido para maximizar o F1 nas previsões de VALIDAÇÃO do treino
    # (cross_val_predict: cada estudante é previsto por um modelo que não o viu no ajuste).
    # O teste nunca é usado para escolher nada.
    escore = cross_val_predict(clone(modelo), X_train, y_train, groups=grupos, cv=cv,
                               method=metodoPontuacao(modelo))
    if escore.ndim == 2:
        escore = escore[:, 1]

    precisao, recall, limiares = precision_recall_curve(y_train, escore)
    f1 = 2 * precisao[:-1] * recall[:-1] / np.clip(precisao[:-1] + recall[:-1], 1e-12, None)
    limiar = float(limiares[np.nanargmax(f1)])

    return limiar, float(average_precision_score(y_train, escore))


# ----------------------------------------------------------------------------------------------------
# AVALIAÇÃO NO TESTE
# ----------------------------------------------------------------------------------------------------

def topoRisco(y, escore, fracao=0.10):
    # Entre os 10% de maior risco: quantos evasores foram capturados (recall) e quantas vezes a taxa de
    # evasão nesse grupo supera a média (lift). É a métrica mais próxima do uso real: só dá para
    # abordar uma parte dos estudantes.
    y = np.asarray(y)
    n = max(1, int(round(len(y) * fracao)))
    indices = np.argsort(-escore)[:n]
    recall = y[indices].sum() / y.sum()
    return recall, recall / fracao


def avaliarModelo(nome, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia do {nome}: {acuracia}")
    return acuracia


def avaliarModeloCompleto(nome, modelo, X_test, y_test, limiar, limiares=()):
    escore = pontuacao(modelo, X_test)
    y = np.asarray(y_test)
    baseline = max(y.mean(), 1 - y.mean())

    print(f"\n=== {nome} ===")
    print(f"Baseline (sempre prever a classe majoritaria): {baseline:.4f} | prevalencia da evasao: {y.mean():.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, escore):.4f} | PR-AUC: {average_precision_score(y, escore):.4f}")
    recall10, lift10 = topoRisco(y, escore)
    print(f"Entre os 10% de maior risco: recall = {recall10:.4f} | lift = {lift10:.2f}")

    resultado = {}
    for rotulo, valor in [("limiar F1 (escolhido no treino)", limiar)] + [(f"limiar {t}", t) for t in limiares]:
        pred = (escore >= valor).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        print(f"\n{rotulo} = {valor:.3f}: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"  Acuracia={accuracy_score(y, pred):.4f} | "
              f"Precisao={precision_score(y, pred, zero_division=0):.4f} | "
              f"Recall={recall_score(y, pred, zero_division=0):.4f} | "
              f"F1={f1_score(y, pred, zero_division=0):.4f}")
        if not resultado:   # o primeiro é o limiar F1: é ele que vai para o resumo
            resultado = {"limiar": valor, "precisao": precision_score(y, pred, zero_division=0),
                         "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
                         "roc_auc": roc_auc_score(y, escore), "pr_auc": average_precision_score(y, escore),
                         "lift_top10": lift10, "recall_top10": recall10}
    return resultado


def resumirModelos(resultados, pr_auc_validacao):
    # Tabela final. O "melhor" modelo é escolhido pela PR-AUC de VALIDAÇÃO (treino), e não pelo teste,
    # para o teste não influenciar a escolha.
    tabela = pd.DataFrame(resultados).T
    tabela.insert(0, "pr_auc_treino", pd.Series(pr_auc_validacao))
    tabela = tabela.sort_values("pr_auc_treino", ascending=False)

    print("\n" + "=" * 100)
    print("RESUMO NO TESTE (ordenado pela PR-AUC de validacao no treino)")
    print("=" * 100)
    print(tabela.round(3).to_string())
    print(f"\nMelhor modelo pelo criterio da validacao: {tabela.index[0]}")

    return tabela


def faixasDeRisco(escore, y_test):
    # Ordena os estudantes do teste pelo escore e divide em 3 faixas de risco.
    y = np.asarray(y_test)
    percentil = pd.Series(escore).rank(pct=True).values
    faixa = pd.cut(percentil, [0, 0.7, 0.9, 1.0], labels=["Baixo (70% menores)", "Medio (70-90%)", "Alto (10% maiores)"],
                   include_lowest=True)
    tabela = pd.DataFrame({"faixa": faixa, "evasao": y}).groupby("faixa", observed=True)["evasao"].agg(["count", "mean"])
    tabela.columns = ["estudantes", "evasao_real"]
    tabela["razao_para_a_media"] = tabela["evasao_real"] / y.mean()

    print("\n--- Evasao real por faixa de risco (teste) ---")
    print(tabela.round(3).to_string())
    return tabela


def calcularMetricas(modelo, X_test, y_test, limiar):
    # Mesmas métricas do avaliarModeloCompleto, mas sem imprimir (usado no experimento de cortes).
    escore = pontuacao(modelo, X_test)
    y = np.asarray(y_test)
    pred = (escore >= limiar).astype(int)
    recall10, lift10 = topoRisco(y, escore)
    pr_auc = average_precision_score(y, escore)
    return {"prevalencia": y.mean(), "roc_auc": roc_auc_score(y, escore), "pr_auc": pr_auc,
            "pr_auc_relativa": pr_auc / y.mean(), "precisao": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "lift_top10": lift10}


def importanciaPermutacao(modelo, X_test, y_test, quantidade=10, amostra=2500, repeticoes=3):
    # Importância por PERMUTAÇÃO: embaralha uma coluna por vez no teste e mede quanto a PR-AUC cai.
    # Quanto maior a queda, mais o modelo depende daquela variável. Funciona para qualquer modelo
    # (inclusive o Voting e o SVM). Usa uma amostra do teste para não demorar. Mostra associação, não causa.
    X_amostra = X_test.sample(min(amostra, len(X_test)), random_state=42)
    y_amostra = np.asarray(y_test)[X_test.index.get_indexer(X_amostra.index)]
    pi = permutation_importance(modelo, X_amostra, y_amostra, scoring="average_precision",
                                n_repeats=repeticoes, random_state=42, n_jobs=1)
    imp = pd.DataFrame({"importancia": pi.importances_mean, "desvio": pi.importances_std},
                       index=X_test.columns).sort_values("importancia", ascending=False)

    print(f"\n--- Importancia por permutacao (queda da PR-AUC no teste, amostra de {len(X_amostra)}) ---")
    print(imp.head(quantidade).round(4).to_string())
    return imp
