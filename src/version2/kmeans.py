import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessamento import reverterLog

# O KMeans é NÃO supervisionado: agrupa estudantes parecidos SEM olhar o alvo (evasao).
# Só depois, para validar, vemos se os grupos encontrados têm taxas de evasão diferentes.
# Usa só variáveis de comportamento (engajamento no VLE + desempenho/entregas + carga).
COLUNAS_CLUSTER = [
    "log_vle_total_cliques", "vle_dias_ativos", "log_vle_cliques_ult_14d", "vle_dias_sem_acesso",
    "vle_tendencia_14d", "aval_n_entregues", "aval_score_medio", "aval_taxa_entrega",
    "aval_n_atrasadas", "creditos_cursados", "num_tentativas_anteriores",
]


def criarPreparoKMeans():
    # O KMeans usa distância Euclidiana: sem padronizar, as colunas de escala grande mandariam sozinhas.
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def compararKMeans(X_train, valores=(2, 3, 4, 5, 6, 7, 8), faixa_escolha=(3, 6), tamanho_amostra=6000):
    # Cotovelo (inércia) + silhueta. k=2 costuma só separar "ativos" de "inativos", então a escolha
    # é o melhor k da silhueta DENTRO da faixa (3 a 6).
    # Silhueta: ~0.5 ou mais = grupos bem separados | abaixo de ~0.25 = estrutura fraca.
    Z = criarPreparoKMeans().fit_transform(X_train[COLUNAS_CLUSTER])

    print("\n------------- Comparando k do KMeans (somente treino) -------------")
    print(f"{'k':<4}{'inercia':>12}{'silhueta':>11}")
    silhuetas = {}
    for k in valores:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        silhuetas[k] = silhouette_score(Z, km.labels_, sample_size=tamanho_amostra, random_state=42)
        print(f"{k:<4}{km.inertia_:>12.0f}{silhuetas[k]:>11.3f}")

    candidatos = {k: s for k, s in silhuetas.items() if faixa_escolha[0] <= k <= faixa_escolha[1]}
    melhor = max(candidatos, key=candidatos.get)
    print(f"melhor k na faixa {faixa_escolha} pela silhueta e {melhor} ({candidatos[melhor]:.3f})")

    return melhor


def treinarKMeans(X_train, k=6):
    preparo = criarPreparoKMeans().fit(X_train[COLUNAS_CLUSTER])
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42).fit(preparo.transform(X_train[COLUNAS_CLUSTER]))

    return preparo, kmeans


def perfilClusters(modelo, X_train, y_train, X_test, y_test):
    # O teste só é ATRIBUÍDO aos grupos já definidos no treino (predict), o KMeans não é reajustado.
    preparo, kmeans = modelo
    cluster_train = kmeans.labels_
    cluster_test = kmeans.predict(preparo.transform(X_test[COLUNAS_CLUSTER]))

    originais = reverterLog(X_train)    # perfis na escala original (cliques de verdade, não log)
    colunas = ["vle_total_cliques", "vle_dias_ativos", "vle_cliques_ult_14d", "vle_dias_sem_acesso",
               "aval_n_entregues", "aval_score_medio", "aval_taxa_entrega", "aval_n_atrasadas",
               "creditos_cursados", "num_tentativas_anteriores"]

    perfil = originais[colunas].groupby(cluster_train).mean().round(2)
    perfil.insert(0, "n_treino", pd.Series(cluster_train).value_counts().sort_index())
    perfil["evasao_treino_%"] = (y_train.groupby(cluster_train).mean() * 100).round(1)
    perfil["evasao_teste_%"] = (y_test.groupby(cluster_test).mean() * 100).round(1)
    perfil["n_teste"] = pd.Series(cluster_test).value_counts().sort_index()
    perfil = perfil.sort_values("evasao_treino_%", ascending=False)

    print("\n--- Perfil dos clusters (ordenado pela evasao no treino; evasao media no teste = "
          f"{y_test.mean() * 100:.1f}%) ---")
    print(perfil.T.to_string())
    return perfil
