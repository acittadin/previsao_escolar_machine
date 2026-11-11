"""
Clusterização não supervisionada (KMeans) de perfis de estudantes.

- Usa apenas variáveis de comportamento (engajamento no VLE + desempenho/entregas + carga).
- O alvo (evasão) NÃO entra no ajuste; ele só é usado depois para validar se os perfis
  encontrados são "de risco" diferentes.
- Ajuste no treino; os estudantes do teste (oferta mais recente) são apenas atribuídos aos clusters.

Uso:  python -m src.clusterizacao
"""
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.limpeza import dividir_treino_teste, reverter_log

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

FEATURES = [
    "log_vle_total_cliques", "vle_dias_ativos", "log_vle_cliques_ult_14d", "vle_dias_sem_acesso",
    "vle_tendencia_14d", "aval_n_entregues", "aval_score_medio", "aval_taxa_entrega",
    "aval_n_atrasadas", "creditos_cursados", "num_tentativas_anteriores",
]
K_MIN, K_MAX = 2, 8          # faixa explorada
K_ESCOLHA_MIN, K_ESCOLHA_MAX = 3, 6   # k = 2 costuma ser trivial (ativos x inativos)


def pre() -> Pipeline:
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])


def main():
    df = pd.read_csv(C.DATASET_LIMPO)
    tr, te = dividir_treino_teste(df)

    prep = pre().fit(tr[FEATURES])
    Xtr, Xte = prep.transform(tr[FEATURES]), prep.transform(te[FEATURES])

    # 1) escolha de k: inércia (cotovelo) + silhueta --------------------------------------------
    linhas = []
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(Xtr)
        sil = silhouette_score(Xtr, km.labels_, sample_size=6000, random_state=C.SEED)
        linhas.append({"k": k, "inercia": km.inertia_, "silhueta": sil})
    sel = pd.DataFrame(linhas)
    sel.to_csv(C.TABELAS / "kmeans_selecao_k.csv", index=False)
    print(sel.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sel["k"], sel["inercia"], "o-")
    axes[0].set(title="Método do cotovelo", xlabel="k", ylabel="Inércia")
    axes[1].plot(sel["k"], sel["silhueta"], "o-", color="#E45756")
    axes[1].set(title="Coeficiente de silhueta", xlabel="k", ylabel="Silhueta")
    fig.tight_layout()
    fig.savefig(C.FIG_CLUSTERS / "01_selecao_k.png", dpi=130)
    plt.close(fig)

    janela = sel[(sel.k >= K_ESCOLHA_MIN) & (sel.k <= K_ESCOLHA_MAX)]
    k = int(janela.loc[janela["silhueta"].idxmax(), "k"])
    print(f"\nk escolhido (melhor silhueta entre {K_ESCOLHA_MIN} e {K_ESCOLHA_MAX}): {k}")

    # 2) modelo final -----------------------------------------------------------------------------
    km = KMeans(n_clusters=k, n_init=20, random_state=C.SEED).fit(Xtr)
    tr["cluster"] = km.labels_
    te["cluster"] = km.predict(Xte)
    joblib.dump({"prep": prep, "kmeans": km, "features": FEATURES}, C.MODELOS / "kmeans.joblib")

    # 3) perfis (escala original) ---------------------------------------------------------------
    todos = pd.concat([tr.assign(base="treino"), te.assign(base="teste")], ignore_index=True)
    orig = reverter_log(todos)
    perfil_cols = ["vle_total_cliques", "vle_dias_ativos", "vle_cliques_ult_14d", "vle_dias_sem_acesso",
                   "aval_n_entregues", "aval_score_medio", "aval_taxa_entrega", "aval_n_atrasadas",
                   "creditos_cursados", "num_tentativas_anteriores"]
    perfil = orig[orig.base == "treino"].groupby("cluster")[perfil_cols].mean().round(2)
    perfil.insert(0, "n_treino", orig[orig.base == "treino"].groupby("cluster").size())
    perfil["evasao_treino_%"] = (orig[orig.base == "treino"].groupby("cluster")[C.ALVO].mean() * 100).round(1)
    perfil["evasao_teste_%"] = (orig[orig.base == "teste"].groupby("cluster")[C.ALVO].mean() * 100).round(1)
    perfil["n_teste"] = orig[orig.base == "teste"].groupby("cluster").size()
    perfil = perfil.sort_values("evasao_treino_%", ascending=False)
    perfil.to_csv(C.TABELAS / "kmeans_perfil_clusters.csv")
    print("\nPerfil dos clusters (ordenado por evasão no treino):\n", perfil.T.to_string())

    todos[[C.COL_ID, "codigo_modulo", C.COL_OFERTA, "base", "cluster", C.ALVO]].to_csv(
        C.TABELAS / "kmeans_atribuicao_clusters.csv", index=False)

    # 4) gráficos -------------------------------------------------------------------------------------
    ordem = perfil.index.tolist()

    # 4a) taxa de evasão por cluster (treino x teste)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ordem))
    ax.bar(x - 0.2, perfil["evasao_treino_%"], 0.4, label="treino", color="#4C78A8")
    ax.bar(x + 0.2, perfil["evasao_teste_%"], 0.4, label=f"teste ({C.OFERTA_TESTE})", color="#E45756")
    ax.axhline(todos[C.ALVO].mean() * 100, color="k", ls="--", lw=1, label="média geral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}\n(n={perfil.loc[c, 'n_treino']})" for c in ordem])
    ax.set(ylabel="Taxa de evasão (%)", title="Evasão por cluster (os clusters NÃO usaram o alvo)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_CLUSTERS / "02_evasao_por_cluster.png", dpi=130)
    plt.close(fig)

    # 4b) heatmap de centróides (z-score)
    cent = pd.DataFrame(km.cluster_centers_, columns=FEATURES).loc[ordem]
    cent.index = [f"Cluster {c}" for c in ordem]
    fig, ax = plt.subplots(figsize=(10, 4 + 0.3 * k))
    sns.heatmap(cent, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Centróides dos clusters (z-score das variáveis)")
    fig.tight_layout()
    fig.savefig(C.FIG_CLUSTERS / "03_centroides_heatmap.png", dpi=130)
    plt.close(fig)

    # 4c) projeção PCA 2D
    pca = PCA(n_components=2, random_state=C.SEED).fit(Xtr)
    idx = np.random.RandomState(C.SEED).choice(len(Xtr), size=min(6000, len(Xtr)), replace=False)
    Z = pca.transform(Xtr[idx])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=tr["cluster"].values[idx], palette="tab10", s=12,
                    alpha=0.6, ax=ax)
    ax.set(title=f"Clusters em 2D (PCA, {pca.explained_variance_ratio_.sum():.0%} da variância)",
           xlabel="PC1", ylabel="PC2")
    ax.legend(title="Cluster")
    fig.tight_layout()
    fig.savefig(C.FIG_CLUSTERS / "04_pca_clusters.png", dpi=130)
    plt.close(fig)

    print(f"\nFiguras em {C.FIG_CLUSTERS}")


if __name__ == "__main__":
    main()
