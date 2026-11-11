"""
Experimento: como a capacidade de prever a evasão muda conforme o dia de corte do curso?

Para cada corte N (dias desde o início da oferta):
  1. agrega os dados usando SOMENTE eventos até o dia N (cache em data/processed/cortes/);
  2. remove quem já cancelou a matrícula até o dia N (a evasão já é conhecida);
  3. limpa, divide (split temporal igual ao experimento principal) e treina os modelos;
  4. avalia no teste (oferta 2014J), com limiar de decisão escolhido por F1 out-of-fold no treino.

Os hiperparâmetros são FIXOS (os escolhidos na busca do corte de 90 dias), para que a comparação
entre cortes isole o efeito da informação disponível. Ver limitações no README.

Uso:
    python -m src.experimento_cortes --agregar 0 30 60          # etapa 1 (lê data/raw)
    python -m src.experimento_cortes --modelar 0 30 60 90       # etapas 2-4
    python -m src.experimento_cortes --relatorio                # tabelas e gráficos finais
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config as C
from src.limpeza import colunas_modelo, dividir_treino_teste, limpar
from src.modelagem import limiar_f1, metodo_score, metricas, pipe, pontuacao

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

CORTES_PADRAO = [0, 30, 60, 90, 120, 180]
PASTA_CORTES = C.DADOS_PROC / "cortes"
ARQ_RESULTADOS = C.TABELAS / "cortes_resultados.csv"
FIG_CORTES = C.RAIZ / "reports" / "figures" / "cortes"
PASTA_CORTES.mkdir(parents=True, exist_ok=True)
FIG_CORTES.mkdir(parents=True, exist_ok=True)


def modelos(num, cat):
    """Mesmos modelos do experimento principal, com os hiperparâmetros escolhidos no corte de 90 dias."""
    S = C.SEED
    return {
        "Regressão Logística (baseline)": pipe(LogisticRegression(
            C=1, class_weight="balanced", max_iter=3000, random_state=S), num, cat, True),
        "Árvore de Decisão": pipe(DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=100, class_weight="balanced", random_state=S), num, cat, False),
        "Random Forest": pipe(RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=30, class_weight="balanced_subsample",
            n_jobs=-1, random_state=S), num, cat, False),
        "Extra Trees": pipe(ExtraTreesClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=S), num, cat, False),
        "Gradient Boosting": pipe(HistGradientBoostingClassifier(
            learning_rate=0.03, max_depth=5, max_iter=200, min_samples_leaf=50, class_weight="balanced",
            early_stopping=False, random_state=S), num, cat, False),
        "SVM RBF": pipe(SVC(
            kernel="rbf", C=0.5, gamma=0.01, class_weight="balanced", cache_size=1000, random_state=S),
            num, cat, True),
    }


# --------------------------------------------------------------------------------------------
def agregar(cortes, pasta_raw: Path):
    from src.agregacao import carregar, montar
    from src.traducao import traduzir

    t0 = time.time()
    d = carregar(pasta_raw)
    print(f"CSVs carregados em {time.time() - t0:.0f}s")
    for n in cortes:
        t1 = time.time()
        df = traduzir(montar(pasta_raw, n, d=d))
        df.to_csv(PASTA_CORTES / f"dataset_corte_{n}.csv", index=False)
        print(f"corte {n:>3}: {df.shape} | evasão {df[C.ALVO].mean():.1%} | {time.time() - t1:.0f}s", flush=True)


def modelar(cortes):
    existentes = pd.read_csv(ARQ_RESULTADOS) if ARQ_RESULTADOS.exists() else pd.DataFrame()
    cv = GroupKFold(n_splits=3)
    for n in cortes:
        arq = PASTA_CORTES / f"dataset_corte_{n}.csv"
        if not arq.exists():
            print(f"corte {n}: dataset não encontrado ({arq.name}); rode --agregar antes")
            continue
        df, _ = limpar(pd.read_csv(arq), cutoff=n)
        tr, te = dividir_treino_teste(df)
        num, cat = colunas_modelo(df)
        feats = num + cat
        Xtr, ytr, gtr = tr[feats], tr[C.ALVO].values, tr[C.COL_ID].values
        Xte, yte = te[feats], te[C.ALVO].values
        print(f"\n=== corte {n} dias | treino {len(tr)} ({ytr.mean():.1%}) | teste {len(te)} ({yte.mean():.1%}) "
              f"| {len(feats)} variáveis", flush=True)

        linhas = []
        for nome, est in modelos(num, cat).items():
            t1 = time.time()
            oof = cross_val_predict(clone(est), Xtr, ytr, groups=gtr, cv=cv, method=metodo_score(est))
            if oof.ndim == 2:
                oof = oof[:, 1]
            lim = limiar_f1(ytr, oof)
            modelo = clone(est).fit(Xtr, ytr)
            s = pontuacao(modelo, Xte)
            linhas.append({
                "corte_dias": n, "modelo": nome, "n_treino": len(tr), "n_teste": len(te),
                "prevalencia_treino": ytr.mean(), "prevalencia_teste": yte.mean(),
                "pr_auc_oof": average_precision_score(ytr, oof), "limiar": lim, **metricas(yte, s, lim),
            })
            r = linhas[-1]
            print(f"  {nome:<32} ROC={r['roc_auc']:.3f} PR={r['pr_auc']:.3f} F1={r['f1']:.3f} "
                  f"lift@10%={r['lift_top10%']:.2f} ({time.time() - t1:.0f}s)", flush=True)

        if len(existentes):
            existentes = existentes[existentes["corte_dias"] != n]
        existentes = pd.concat([existentes, pd.DataFrame(linhas)], ignore_index=True)
        existentes.sort_values(["corte_dias", "modelo"]).round(4).to_csv(ARQ_RESULTADOS, index=False)


def relatorio():
    r = pd.read_csv(ARQ_RESULTADOS)
    # PR-AUC dividida pela prevalência = "quantas vezes melhor que o acaso" (comparável entre cortes)
    r["pr_auc_relativa"] = r["pr_auc"] / r["prevalencia_teste"]
    cortes = sorted(r["corte_dias"].unique())
    print(f"Cortes disponíveis: {cortes}")

    prev = r.groupby("corte_dias")[["n_treino", "n_teste", "prevalencia_treino", "prevalencia_teste"]].first()
    prev.round(4).to_csv(C.TABELAS / "cortes_populacao.csv")
    print("\nPopulação por corte:\n", prev.round(3).to_string())

    resumo = r.pivot(index="modelo", columns="corte_dias", values="pr_auc").round(3)
    print("\nPR-AUC no teste:\n", resumo.to_string())
    print("\nPR-AUC relativa ao acaso (PR-AUC / prevalência):\n",
          r.pivot(index="modelo", columns="corte_dias", values="pr_auc_relativa").round(2).to_string())
    print("\nROC-AUC no teste:\n", r.pivot(index="modelo", columns="corte_dias", values="roc_auc").round(3).to_string())
    print("\nLift no top 10%:\n", r.pivot(index="modelo", columns="corte_dias", values="lift_top10%").round(2).to_string())
    print("\nF1 (limiar do treino):\n", r.pivot(index="modelo", columns="corte_dias", values="f1").round(3).to_string())

    # melhor modelo por corte, escolhido pela PR-AUC out-of-fold (critério do treino)
    melhores = r.loc[r.groupby("corte_dias")["pr_auc_oof"].idxmax(),
                     ["corte_dias", "modelo", "pr_auc_oof", "roc_auc", "pr_auc", "f1", "recall", "precisao",
                      "lift_top10%", "recall_top10%"]]
    melhores.round(3).to_csv(C.TABELAS / "cortes_melhor_modelo.csv", index=False)
    print("\nMelhor modelo por corte (critério: PR-AUC out-of-fold):\n", melhores.round(3).to_string(index=False))

    estilo = dict(marker="o")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    paineis = [("roc_auc", "ROC-AUC"), ("pr_auc_relativa", "PR-AUC ÷ prevalência (x vezes o acaso)"),
               ("lift_top10%", "Lift entre os 10% de maior risco"), ("pr_auc", "PR-AUC (NÃO comparável entre cortes)"),
               ("f1", "F1 (NÃO comparável entre cortes)"), ("recall_top10%", "Recall entre os 10% de maior risco")]
    for ax, (col, tit) in zip(axes.ravel(), paineis):
        sns.lineplot(data=r, x="corte_dias", y=col, hue="modelo", ax=ax, **estilo)
        ax.set(title=tit, xlabel="Dia de corte do curso", ylabel="")
        ax.set_xticks(cortes)
        if col == "pr_auc":
            ax.plot(prev.index, prev["prevalencia_teste"], "k--", label="acaso (prevalência)")
        if col in ("pr_auc_relativa", "roc_auc"):
            ax.axhline(1 if col == "pr_auc_relativa" else 0.5, color="k", ls="--", lw=1)
        ax.legend(fontsize=6)
    fig.suptitle(f"Desempenho no teste ({C.OFERTA_TESTE}) conforme o dia de corte")
    fig.tight_layout()
    fig.savefig(FIG_CORTES / "01_desempenho_por_corte.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(prev.index, prev["prevalencia_teste"] * 100, "o-", color="#E45756", label="teste")
    ax.plot(prev.index, prev["prevalencia_treino"] * 100, "o--", color="#4C78A8", label="treino")
    ax.set(xlabel="Dia de corte", ylabel="Taxa de evasão restante (%)",
           title="Evasão ainda por acontecer entre os estudantes ativos no corte")
    ax.set_xticks(cortes)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_CORTES / "02_prevalencia_por_corte.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([str(c) for c in prev.index], prev["n_teste"], color="#4C78A8")
    ax.set(xlabel="Dia de corte", ylabel="Estudantes no teste", title="Estudantes ainda ativos no corte (teste)")
    fig.tight_layout()
    fig.savefig(FIG_CORTES / "03_estudantes_ativos_por_corte.png", dpi=130)
    plt.close(fig)
    print(f"\nFiguras em {FIG_CORTES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agregar", nargs="*", type=int, help="cortes a agregar (usa data/raw)")
    ap.add_argument("--modelar", nargs="*", type=int, help="cortes a modelar")
    ap.add_argument("--relatorio", action="store_true")
    ap.add_argument("--raw", default=str(C.DADOS_RAW))
    a = ap.parse_args()
    if a.agregar is not None:
        agregar(a.agregar or CORTES_PADRAO, Path(a.raw))
    if a.modelar is not None:
        modelar(a.modelar or CORTES_PADRAO)
    if a.relatorio:
        relatorio()
