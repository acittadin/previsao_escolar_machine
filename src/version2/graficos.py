from pathlib import Path

import matplotlib
matplotlib.use("Agg")      # só salva em arquivo, não abre janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from preprocessamento import ALVO, COL_ID

# Gera as figuras do relatório em reports/figures/. Cada função salva um PNG (150 dpi).
sns.set_theme(style="whitegrid")
PASTA = Path("reports/figures")

NOMES_BONITOS = {
    "aval_score_medio": "Nota média", "aval_taxa_entrega": "Taxa de entrega", "aval_score_min": "Nota mínima",
    "aval_score_ponderado_soma": "Nota ponderada acumulada", "vle_cliques_ult_14d": "Cliques nos últimos 14 dias",
    "vle_cliques_ult_30d": "Cliques nos últimos 30 dias", "aval_n_entregues": "Avaliações entregues",
    "vle_dias_ativos": "Dias ativos no VLE", "vle_total_cliques": "Total de cliques",
    "vle_sites_distintos": "Sites distintos no VLE", "vle_dias_sem_acesso": "Dias sem acesso",
    "aval_atraso_medio": "Atraso médio nas entregas", "aval_n_atrasadas": "Entregas atrasadas",
    "creditos_cursados": "Créditos cursados", "aval_n_devidas": "Avaliações devidas",
    "vle_cliques_quiz_externo": "Cliques em quiz externo", "vle_tendencia_14d": "Tendência dos últimos 14 dias",
    "num_tentativas_anteriores": "Tentativas anteriores", "codigo_modulo": "Módulo",
    "log_vle_cliques_ult_14d": "Cliques nos últimos 14 dias (log)", "log_vle_total_cliques": "Total de cliques (log)",
    "deficiencia_sim": "Deficiência declarada", "aval_n_banked": "Avaliações aproveitadas",
}


def _salvar(fig, nome):
    PASTA.mkdir(parents=True, exist_ok=True)
    caminho = PASTA / nome
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"figura salva: {caminho}")


def _rotulo(coluna):
    return NOMES_BONITOS.get(coluna, coluna.replace("_", " "))


# ----------------------------------------------------------------------------------------------------
# Análise exploratória (dataset bruto)
# ----------------------------------------------------------------------------------------------------

def graficoResultadoFinal(df):
    ordem = ["Aprovado", "Aprovado com distinção", "Reprovado", "Evadido"]
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    sns.countplot(data=df, x="resultado_final", order=ordem, ax=ax, color="#4C78A8")
    ax.patches[3].set_color("#E45756")
    ax.set(title="Resultado final (estudantes ativos no dia 90)", xlabel="", ylabel="Estudantes")
    plt.setp(ax.get_xticklabels(), rotation=15)
    _salvar(fig, "01_resultado_final.png")


def graficoCorrelacao(df, quantidade=7):
    numericas = df.select_dtypes("number").drop(columns=[COL_ID, ALVO, "vle_ultimo_acesso"], errors="ignore")
    corr = numericas.apply(lambda s: s.corr(df[ALVO], method="spearman")).dropna().sort_values()
    corr = pd.concat([corr.head(quantidade), corr.tail(quantidade)])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh([_rotulo(c) for c in corr.index], corr.values, color=["#E45756" if v > 0 else "#4C78A8" for v in corr])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Correlação de Spearman com a evasão")
    ax.tick_params(labelsize=10)
    _salvar(fig, "02_correlacao_evasao.png")


def graficoTaxaModulo(df):
    tabela = df.groupby("codigo_modulo")[ALVO].agg(["mean", "count"]).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=tabela.index, y=tabela["mean"] * 100, color="#E45756", ax=ax)
    ax.axhline(df[ALVO].mean() * 100, color="k", ls="--", lw=1, label="média geral")
    for i, n in enumerate(tabela["count"]):
        ax.text(i, tabela["mean"].iloc[i] * 100 + 0.3, f"n={n}", ha="center", fontsize=8)
    ax.set(ylabel="Taxa de evasão (%)", xlabel="", title="Taxa de evasão por módulo")
    ax.legend()
    _salvar(fig, "03_taxa_evasao_modulo.png")


# ----------------------------------------------------------------------------------------------------
# KMeans
# ----------------------------------------------------------------------------------------------------

def graficoClusters(perfil, evasao_media_teste):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(perfil))
    ax.bar(x - 0.2, perfil["evasao_treino_%"], 0.4, label="treino", color="#4C78A8")
    ax.bar(x + 0.2, perfil["evasao_teste_%"], 0.4, label="teste (2014J)", color="#E45756")
    ax.axhline(evasao_media_teste * 100, color="k", ls="--", lw=1, label="média do teste")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}\n(n={int(perfil.loc[c, 'n_treino'])})" for c in perfil.index])
    ax.set(ylabel="Taxa de evasão (%)", title="Evasão por cluster (o KMeans não usou o alvo)")
    ax.legend()
    _salvar(fig, "04_evasao_por_cluster.png")


# ----------------------------------------------------------------------------------------------------
# Modelos (teste)
# ----------------------------------------------------------------------------------------------------

def graficoROC(escores, y_test, ordem):
    fig, ax = plt.subplots(figsize=(7, 6))
    for nome in ordem:
        fpr, tpr, _ = roc_curve(y_test, escores[nome])
        ax.plot(fpr, tpr, label=f"{nome} ({roc_auc_score(y_test, escores[nome]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", title="Curvas ROC no teste")
    ax.legend(fontsize=8, loc="lower right")
    _salvar(fig, "05_roc.png")


def graficoPrecisaoRecall(escores, y_test, ordem):
    fig, ax = plt.subplots(figsize=(7, 6))
    for nome in ordem:
        p, r, _ = precision_recall_curve(y_test, escores[nome])
        ax.plot(r, p, label=f"{nome} ({average_precision_score(y_test, escores[nome]):.3f})")
    ax.axhline(np.mean(y_test), color="k", ls="--", lw=1, label=f"acaso ({np.mean(y_test):.3f})")
    ax.set(xlabel="Recall", ylabel="Precisão", title="Curvas Precisão-Recall no teste")
    ax.legend(fontsize=8)
    _salvar(fig, "06_precisao_recall.png")


def graficoMatrizConfusao(nome, escore, y_test, limiar):
    y = np.asarray(y_test)
    pred = (escore >= limiar).astype(int)
    cm = np.array([[((pred == 0) & (y == 0)).sum(), ((pred == 1) & (y == 0)).sum()],
                   [((pred == 0) & (y == 1)).sum(), ((pred == 1) & (y == 1)).sum()]])
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 14},
                xticklabels=["permanece", "evade"], yticklabels=["permanece", "evade"])
    ax.set(xlabel="Previsto", ylabel="Real", title=f"{nome}, limiar {limiar:.3f}")
    ax.title.set_fontsize(11)
    _salvar(fig, "07_matriz_confusao.png")


def graficoImportancia(importancias, nome, quantidade=12):
    top = importancias.head(quantidade).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh([_rotulo(c) for c in top.index], top["importancia"], xerr=top["desvio"], color="#4C78A8", capsize=2)
    ax.set_xlabel("Queda média da PR-AUC (3 repetições)")
    ax.tick_params(labelsize=10)
    _salvar(fig, "08_importancia_permutacao.png")


# ----------------------------------------------------------------------------------------------------
# Experimento de cortes
# ----------------------------------------------------------------------------------------------------

def graficoCortes(resultados):
    # PR-AUC absoluta não entra: cai com o corte só porque a evasão restante diminui (não é comparável).
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.3))
    for nome, g in resultados.groupby("modelo"):
        g = g.sort_values("corte")
        eixos[0].plot(g["corte"], g["roc_auc"], "o-", label=nome, ms=4)
        eixos[1].plot(g["corte"], g["pr_auc_relativa"], "o-", label=nome, ms=4)
    eixos[0].axhline(0.5, color="k", ls="--", lw=1)
    eixos[1].axhline(1, color="k", ls="--", lw=1)
    eixos[0].set(title="ROC-AUC no teste", xlabel="Dia de corte do curso")
    eixos[1].set(title="PR-AUC ÷ prevalência (vezes o acaso)", xlabel="Dia de corte do curso")
    for ax in eixos:
        ax.set_xticks(sorted(resultados["corte"].unique()))
        ax.tick_params(labelsize=10)
    h, l = eixos[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    PASTA.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA / "09_desempenho_por_corte.png", dpi=150)
    plt.close(fig)
    print(f"figura salva: {PASTA / '09_desempenho_por_corte.png'}")


def gerarGraficosDados(caminho):
    df = pd.read_csv(caminho)
    graficoResultadoFinal(df)
    graficoCorrelacao(df)
    graficoTaxaModulo(df)
