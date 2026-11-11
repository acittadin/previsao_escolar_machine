"""
Análise exploratória (EDA) do dataset agregado e traduzido.
Gera figuras em reports/figures/eda e tabelas em reports/tables.

Uso:  python -m src.eda
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as C

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
COR = {0: "#4C78A8", 1: "#E45756"}
LABEL = {0: "permanece", 1: "evade"}


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(C.FIG_EDA / nome, dpi=130)
    plt.close(fig)


def taxa_por(df, col, ordem=None):
    t = df.groupby(col)[C.ALVO].agg(taxa_evasao="mean", n="count")
    return t.reindex(ordem) if ordem else t.sort_values("taxa_evasao", ascending=False)


def grafico_taxa(df, col, nome, ordem=None, rot=0):
    t = taxa_por(df, col, ordem)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=t.index.astype(str), y=t["taxa_evasao"] * 100, color="#E45756", ax=ax)
    ax.axhline(df[C.ALVO].mean() * 100, color="k", ls="--", lw=1, label="média geral")
    for i, n in enumerate(t["n"]):
        ax.text(i, t["taxa_evasao"].iloc[i] * 100 + 0.3, f"n={n}", ha="center", fontsize=8)
    ax.set(ylabel="Taxa de evasão (%)", xlabel="", title=f"Taxa de evasão por {col}")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="right" if rot else "center")
    salvar(fig, nome)
    return t


def main():
    df = pd.read_csv(C.DATASET_PT)
    tabelas = {}

    # 1) visão geral ------------------------------------------------------------------------
    print(f"Shape: {df.shape} | estudantes únicos: {df[C.COL_ID].nunique()} | "
          f"taxa de evasão: {df[C.ALVO].mean():.1%}")
    resumo = pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "nulos_%": (df.isna().mean() * 100).round(2),
        "unicos": df.nunique(),
    })
    resumo.to_csv(C.TABELAS / "eda_resumo_colunas.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    ordem = ["Aprovado", "Aprovado com distinção", "Reprovado", "Evadido"]
    sns.countplot(data=df, x="resultado_final", order=ordem, ax=ax, color="#4C78A8")
    ax.patches[3].set_color("#E45756") if len(ax.patches) > 3 else None
    ax.set(title="Resultado final (estudantes ativos no corte)", xlabel="")
    plt.setp(ax.get_xticklabels(), rotation=15)
    salvar(fig, "01_resultado_final.png")

    # 2) nulos --------------------------------------------------------------------------------
    nulos = (df.isna().mean() * 100)
    nulos = nulos[nulos > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    nulos.plot.barh(ax=ax, color="#F58518")
    ax.set(xlabel="% de nulos", title="Valores ausentes por coluna")
    salvar(fig, "02_nulos.png")

    # 3) categóricas --------------------------------------------------------------------------
    cats = {
        "escolaridade_max": ["Sem qualificação formal", "Abaixo do A Level", "A Level ou equivalente",
                             "Ensino superior", "Pós-graduação"],
        "faixa_imd": ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%",
                      "80-90%", "90-100%"],
        "faixa_etaria": ["Até 35", "35 a 55", "55 ou mais"],
        "genero": None, "deficiencia": None, "num_tentativas_anteriores": None,
        "codigo_modulo": None, "codigo_oferta": None, "regiao": None,
    }
    for i, (col, ordem) in enumerate(cats.items(), start=3):
        rot = 30 if col in ("escolaridade_max", "regiao", "faixa_imd") else 0
        t = grafico_taxa(df, col, f"{i:02d}_taxa_{col}.png", ordem, rot)
        tabelas[col] = t.assign(taxa_evasao=(t.taxa_evasao * 100).round(1))
    pd.concat(tabelas, names=["variavel", "categoria"]).to_csv(C.TABELAS / "eda_taxa_evasao_categoricas.csv")

    # 4) numéricas ----------------------------------------------------------------------------
    chave = ["vle_total_cliques", "vle_dias_ativos", "vle_cliques_ult_14d", "vle_dias_sem_acesso",
             "aval_n_entregues", "aval_score_medio", "aval_taxa_entrega", "creditos_cursados",
             "data_registro"]
    comp = df.groupby(C.ALVO)[chave].agg(["mean", "median"]).T.round(2)
    comp.to_csv(C.TABELAS / "eda_numericas_por_classe.csv")
    print("\nMédia/mediana por classe:\n", comp.to_string())

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.ravel(), chave):
        x = df[[col, C.ALVO]].dropna()
        x = x[x[col] <= x[col].quantile(0.99)]
        for k in (0, 1):
            sns.kdeplot(x[x[C.ALVO] == k][col], ax=ax, color=COR[k], fill=True, alpha=0.3,
                        label=LABEL[k], common_norm=False)
        ax.set(title=col, xlabel="")
    axes[0, 0].legend()
    salvar(fig, "12_distribuicoes_numericas.png")

    # 5) engajamento --------------------------------------------------------------------------
    df["decil_cliques"] = pd.qcut(df["vle_total_cliques"], 10, labels=[f"D{i}" for i in range(1, 11)],
                                  duplicates="drop")
    t = df.groupby("decil_cliques", observed=True)[C.ALVO].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(t.index.astype(str), t.values, color="#E45756")
    ax.set(xlabel="Decil de cliques totais (D1 = menos engajados)", ylabel="Taxa de evasão (%)",
           title="Evasão por decil de engajamento no VLE")
    salvar(fig, "13_evasao_por_decil_cliques.png")
    t.round(1).to_csv(C.TABELAS / "eda_evasao_por_decil_cliques.csv")

    cols_tipo = [c for c in df.columns if c.startswith("vle_cliques_") and c not in
                 ("vle_cliques_pre_curso", "vle_cliques_ult_14d", "vle_cliques_ult_30d")]
    med = df.groupby(C.ALVO)[cols_tipo].mean().T
    med["razao_evade_permanece"] = med[1] / med[0].replace(0, np.nan)
    med = med.sort_values("razao_evade_permanece")
    med.to_csv(C.TABELAS / "eda_cliques_por_tipo_atividade.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([c.replace("vle_cliques_", "") for c in med.index], med["razao_evade_permanece"],
            color="#4C78A8")
    ax.axvline(1, color="k", ls="--")
    ax.set_title("Cliques médios: evade ÷ permanece (<1 = evasores clicam menos)")
    salvar(fig, "14_razao_tipo_atividade.png")

    # 6) correlação ---------------------------------------------------------------------------
    num = df.select_dtypes("number").drop(columns=[C.COL_ID], errors="ignore")
    corr_alvo = num.drop(columns=[C.ALVO]).apply(lambda s: s.corr(df[C.ALVO], method="spearman"))
    corr_alvo = corr_alvo.dropna().sort_values()
    corr_alvo.round(3).to_csv(C.TABELAS / "eda_correlacao_com_alvo.csv", header=["spearman"])
    fig, ax = plt.subplots(figsize=(8, 9))
    corr_alvo.plot.barh(ax=ax, color=["#E45756" if v > 0 else "#4C78A8" for v in corr_alvo])
    ax.set_title("Correlação de Spearman com a evasão")
    salvar(fig, "15_correlacao_alvo.png")

    base = ["vle_total_cliques", "vle_dias_ativos", "vle_sites_distintos", "vle_cliques_ult_14d",
            "vle_dias_sem_acesso", "aval_n_entregues", "aval_score_medio", "aval_taxa_entrega",
            "aval_n_atrasadas", "creditos_cursados", "num_tentativas_anteriores", C.ALVO]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[base].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, annot_kws={"size": 7})
    ax.set_title("Matriz de correlação (Spearman)")
    salvar(fig, "16_matriz_correlacao.png")

    c = df[[x for x in num.columns if x.startswith(("vle_", "aval_"))]].corr(method="spearman").abs()
    pares = c.where(np.triu(np.ones(c.shape), 1).astype(bool)).stack().sort_values(ascending=False)
    pares[pares > 0.9].round(3).to_csv(C.TABELAS / "eda_pares_redundantes.csv", header=["abs_spearman"])

    print("\nTop correlações com a evasão:\n", pd.concat([corr_alvo.head(5), corr_alvo.tail(5)]).round(3).to_string())
    print(f"\nFiguras em {C.FIG_EDA} | tabelas em {C.TABELAS}")


if __name__ == "__main__":
    main()
