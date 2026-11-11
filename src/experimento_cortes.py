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
from src.limpeza import colunas_modelo, dividir_treino_teste, limpar_dados
from src.modelagem import limiar_f1, metodo_score, metricas, pipe, pontuacao

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

CORTES_PADRAO = [0, 30, 60, 90, 120, 180]
PASTA_CORTES = C.DADOS_PROC / "cortes"
ARQ_RESULTADOS = C.TABELAS / "cortes_resultados.csv"
ARQ_POPULACAO = C.TABELAS / "cortes_populacao.csv"
ARQ_MELHOR = C.TABELAS / "cortes_melhor_modelo.csv"
FIG_CORTES = C.RAIZ / "reports" / "figures" / "cortes"

PASTA_CORTES.mkdir(parents=True, exist_ok=True)
FIG_CORTES.mkdir(parents=True, exist_ok=True)


def instanciar_modelos_fixos(num, cat):
    """
    Retorna os mesmos modelos do experimento principal com hiperparâmetros já
    otimizados no corte de 90 dias, para isolar o efeito da linha do tempo.
    """
    S = C.SEED
    return {
        "Regressão Logística (baseline)": pipe(
            LogisticRegression(C=1, class_weight="balanced", max_iter=3000, random_state=S), num, cat, True
        ),
        "Árvore de Decisão": pipe(
            DecisionTreeClassifier(max_depth=10, min_samples_leaf=100, class_weight="balanced", random_state=S), num, cat, False
        ),
        "Random Forest": pipe(
            RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=30, class_weight="balanced_subsample", n_jobs=-1, random_state=S), num, cat, False
        ),
        "Extra Trees": pipe(
            ExtraTreesClassifier(n_estimators=200, max_depth=14, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=S), num, cat, False
        ),
        "Gradient Boosting": pipe(
            HistGradientBoostingClassifier(learning_rate=0.03, max_depth=5, max_iter=200, min_samples_leaf=50, class_weight="balanced", early_stopping=False, random_state=S), num, cat, False
        ),
        "SVM RBF": pipe(
            SVC(kernel="rbf", C=0.5, gamma=0.01, class_weight="balanced", cache_size=1000, random_state=S), num, cat, True
        ),
    }


def agregar_cortes(cortes, pasta_raw: Path):
    from src.agregacao import carregar_dados_brutos, montar_dataset_final
    from src.traducao import traduzir

    t0 = time.time()
    print("Carregando CSVs originais do OULAD...")
    dados_brutos = carregar_dados_brutos(pasta_raw)
    print(f"Concluído em {time.time() - t0:.0f}s")

    for dias in cortes:
        t1 = time.time()
        df_corte = traduzir(montar_dataset_final(pasta_raw, dias, d=dados_brutos))
        caminho_salvamento = PASTA_CORTES / f"dataset_corte_{dias}.csv"
        df_corte.to_csv(caminho_salvamento, index=False)
        print(f"Corte {dias:>3} dias: {len(df_corte)} registros salvos em {caminho_salvamento.name} ({time.time() - t1:.0f}s)")


def modelar(cortes=None):
    if cortes is None:
        cortes = CORTES_PADRAO

    resultados, resumo_populacao = [], []
    cv = GroupKFold(n_splits=3)

    for dias in cortes:
        caminho = PASTA_CORTES / f"dataset_corte_{dias}.csv"
        if not caminho.exists():
            print(f"Arquivo ausente para corte {dias}: {caminho}. Rode `python main.py --etapas cortes` ou gere os cortes primeiro.")
            continue

        df = pd.read_csv(caminho)
        df_limpo = limpar_dados(df, cutoff=dias)
        tr, te = dividir_treino_teste(df_limpo)
        num, cat = colunas_modelo(df_limpo)
        feats = num + cat

        Xtr, ytr, gtr = tr[feats], tr[C.ALVO].values, tr[C.COL_ID].values
        Xte, yte = te[feats], te[C.ALVO].values

        resumo_populacao.append({
            "corte_dias": dias,
            "n_treino": len(tr),
            "n_teste": len(te),
            "prevalencia_treino": float(ytr.mean()),
            "prevalencia_teste": float(yte.mean()),
        })

        modelos = instanciar_modelos_fixos(num, cat)
        for nome, modelo in modelos.items():
            oof = cross_val_predict(clone(modelo), Xtr, ytr, groups=gtr, cv=cv,
                                   method=metodo_score(modelo), n_jobs=1)
            if oof.ndim == 2:
                oof = oof[:, 1]
            limiar = limiar_f1(ytr, oof)

            modelo_final = clone(modelo).fit(Xtr, ytr)
            s_te = pontuacao(modelo_final, Xte)
            m = metricas(yte, s_te, limiar)
            res = {
                "corte_dias": dias,
                "modelo": nome,
                "n_treino": len(tr),
                "n_teste": len(te),
                "prevalencia_treino": float(ytr.mean()),
                "prevalencia_teste": float(yte.mean()),
                "pr_auc_oof": float(average_precision_score(ytr, oof)),
                "limiar": float(limiar),
                **m,
            }
            resultados.append(res)
            print(f"[{dias:>3}d] {nome:<34} OOF PR-AUC={res['pr_auc_oof']:.3f} | ROC={res['roc_auc']:.3f} PR={res['pr_auc']:.3f} F1={res['f1']:.3f}")

    pd.DataFrame(resultados).sort_values(["corte_dias", "pr_auc_oof"], ascending=[True, False]).to_csv(ARQ_RESULTADOS, index=False)
    pd.DataFrame(resumo_populacao).to_csv(ARQ_POPULACAO, index=False)

    melhor_por_corte = (
        pd.DataFrame(resultados)
        .sort_values(["corte_dias", "pr_auc_oof"], ascending=[True, False])
        .groupby("corte_dias", as_index=False)
        .first()
    )
    melhor_por_corte[["corte_dias", "modelo", "pr_auc_oof", "roc_auc", "pr_auc", "f1", "recall", "precisao", "lift_top10%", "recall_top10%"]].to_csv(ARQ_MELHOR, index=False)

    print(f"\nResultados salvos em {ARQ_RESULTADOS}")
    print(f"População por corte em {ARQ_POPULACAO}")
    print(f"Melhor modelo por corte em {ARQ_MELHOR}")


def relatorio():
    if not ARQ_RESULTADOS.exists():
        print(f"Arquivo de resultados ausente em {ARQ_RESULTADOS}. Rode modelar() primeiro.")
        return

    df = pd.read_csv(ARQ_RESULTADOS)
    if df.empty:
        print("Nenhum resultado disponível para o relatório.")
        return

    melhor = (df.sort_values(["corte_dias", "pr_auc_oof"], ascending=[True, False])
               .groupby("corte_dias", as_index=False)
               .first())
    print("\nMelhor modelo por corte (critério: PR-AUC out-of-fold no treino):")
    print(melhor[["corte_dias", "modelo", "pr_auc_oof", "roc_auc", "pr_auc", "lift_top10%"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    for nome in sorted(df["modelo"].unique()):
        tmp = df[df["modelo"] == nome].sort_values("corte_dias")
        ax.plot(tmp["corte_dias"], tmp["roc_auc"], marker="o", label=nome)
    ax.set_title("ROC-AUC por corte de tempo")
    ax.set_xlabel("Dias de corte")
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_CORTES / "roc_auc_por_corte.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Experimento com cortes temporais para estimar atrelamento do desempenho ao tempo no curso.")
    parser.add_argument("--cortes", nargs="*", type=int, default=CORTES_PADRAO)
    parser.add_argument("--agregar", action="store_true")
    args = parser.parse_args()

    if args.agregar:
        agregar_cortes(args.cortes, C.DADOS_RAW)
    modelar(args.cortes)
    relatorio()


if __name__ == "__main__":
    main()
