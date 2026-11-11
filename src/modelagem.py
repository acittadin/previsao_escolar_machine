import argparse
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from src import config as C
from src.limpeza import colunas_modelo, dividir_treino_teste

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def preprocessador(num, cat, escalar: bool) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler() if escalar else "passthrough"),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def pipe(est, num, cat, escalar):
    return Pipeline([("pre", preprocessador(num, cat, escalar)), ("clf", est)])


def especificacoes(num, cat, rapido: bool):
    n_arv = 80 if rapido else 200
    S = C.SEED
    g = (lambda full, small: small if rapido else full)
    return [
        ("Regressão Logística (baseline)", "Baseline",
         pipe(LogisticRegression(class_weight="balanced", max_iter=3000, random_state=S), num, cat, True),
         {"clf__C": g([0.01, 0.1, 1], [0.1])}, None),

        ("Árvore de Decisão", "Árvore",
         pipe(DecisionTreeClassifier(class_weight="balanced", random_state=S), num, cat, False),
         {"clf__max_depth": g([3, 4, 5, 7, 10], [4, 6]), "clf__min_samples_leaf": g([20, 50, 100], [50])}, None),

        ("Random Forest", "Ensemble",
         pipe(RandomForestClassifier(n_estimators=n_arv, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=S), num, cat, False),
         {"clf__max_depth": g([6, 12], [8]), "clf__min_samples_leaf": g([10, 30], [20])}, None),

        ("Extra Trees", "Ensemble",
         pipe(ExtraTreesClassifier(n_estimators=n_arv, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=S), num, cat, False),
         {"clf__max_depth": g([8, 14], [10]), "clf__min_samples_leaf": g([5, 20], [10])}, None),

        ("Gradient Boosting", "Ensemble",
         pipe(HistGradientBoostingClassifier(class_weight="balanced", max_iter=120 if rapido else 200,
                                             early_stopping=False, min_samples_leaf=50, random_state=S),
              num, cat, False),
         {"clf__learning_rate": g([0.03, 0.1], [0.05]), "clf__max_depth": g([3, 5], [3])}, None),

        ("SVM Linear", "SVM",
         pipe(LinearSVC(class_weight="balanced", max_iter=5000, dual=False, random_state=S), num, cat, True),
         {"clf__C": g([0.01, 0.1, 1], [0.1])}, None),

        ("SVM RBF", "SVM",
         pipe(SVC(kernel="rbf", class_weight="balanced", cache_size=1000, random_state=S), num, cat, True),
         {"clf__C": g([0.5, 2], [1]), "clf__gamma": g(["scale", 0.01], ["scale"])},
         4000 if rapido else 6000),
    ]


def pontuacao(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def metodo_score(est):
    return "predict_proba" if hasattr(est, "predict_proba") else "decision_function"


def limiar_f1(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.nanargmax(f1)])


def topo_k(y, s, frac=0.10):
    n = max(1, int(round(len(y) * frac)))
    idx = np.argsort(-s)[:n]
    rec = y[idx].sum() / y.sum()
    return rec, rec / frac


def metricas(y, s, lim):
    pred = (s >= lim).astype(int)
    rec10, lift10 = topo_k(y, s)
    return {
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "precisao": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "acuracia_balanceada": balanced_accuracy_score(y, pred),
        "acuracia": accuracy_score(y, pred),
        "recall_top10%": rec10,
        "lift_top10%": lift10,
    }


def main(rapido: bool = False):
    t0 = time.time()
    df = pd.read_csv(C.DATASET_LIMPO)
    tr, te = dividir_treino_teste(df)
    num, cat = colunas_modelo(df)
    feats = num + cat
    Xtr, ytr, gtr = tr[feats], tr[C.ALVO].values, tr[C.COL_ID].values
    Xte, yte = te[feats], te[C.ALVO].values
    print(f"Treino: {len(tr)} ({ytr.mean():.1%} evasão) | Teste {C.OFERTA_TESTE}: {len(te)} "
          f"({yte.mean():.1%}) | {len(feats)} variáveis | baseline PR-AUC = prevalência = {yte.mean():.3f}")

    cv = GroupKFold(n_splits=3)
    resultados, melhores, ajustados, scores_te, params_tab, oof_scores = [], {}, {}, {}, [], {}

    def avaliar(nome, familia, estimador, cv_best=np.nan, params=None):
        t1 = time.time()
        oof = cross_val_predict(clone(estimador), Xtr, ytr, groups=gtr, cv=cv,
                                method=metodo_score(estimador), n_jobs=1)
        if oof.ndim == 2:
            oof = oof[:, 1]
        lim = limiar_f1(ytr, oof)
        modelo = clone(estimador).fit(Xtr, ytr)
        s_te = pontuacao(modelo, Xte)
        m = metricas(yte, s_te, lim)
        resultados.append({
            "modelo": nome, "familia": familia,
            "pr_auc_cv": cv_best, "pr_auc_oof": average_precision_score(ytr, oof),
            "limiar": lim, **m, "tempo_s": round(time.time() - t1, 1),
        })
        ajustados[nome], scores_te[nome], oof_scores[nome] = modelo, s_te, oof
        params_tab.append({"modelo": nome, "hiperparametros": str(params or {})})
        r = resultados[-1]
        print(f"  {nome:<34} OOF PR-AUC={r['pr_auc_oof']:.3f} | TESTE ROC={r['roc_auc']:.3f} "
              f"PR={r['pr_auc']:.3f} F1={r['f1']:.3f} rec={r['recall']:.3f} prec={r['precisao']:.3f} "
              f"({r['tempo_s']}s)")

    print("\nTreinando modelos...")
    for nome, familia, estimador, grade, subamostra in especificacoes(num, cat, rapido):
        gs = GridSearchCV(estimador, grade, scoring="average_precision", cv=cv, n_jobs=1, refit=False)
        if subamostra and subamostra < len(Xtr):
            idx = np.random.RandomState(C.SEED).choice(len(Xtr), subamostra, replace=False)
            gs.fit(Xtr.iloc[idx], ytr[idx], groups=gtr[idx])
        else:
            gs.fit(Xtr, ytr, groups=gtr)
        final = clone(estimador).set_params(**gs.best_params_)
        melhores[nome] = final
        avaliar(nome, familia, final, gs.best_score_, gs.best_params_)

    voting = VotingClassifier(
        estimators=[("rf", clone(melhores["Random Forest"])),
                    ("gb", clone(melhores["Gradient Boosting"])),
                    ("lr", clone(melhores["Regressão Logística (baseline)"]))],
        voting="soft", n_jobs=1)
    avaliar("Voting (RF + GB + LR)", "Ensemble", voting, np.nan, {"voting": "soft"})

    res = pd.DataFrame(resultados).sort_values("pr_auc_oof", ascending=False).reset_index(drop=True)
    res.round(4).to_csv(C.TABELAS / "comparacao_modelos.csv", index=False)
    pd.DataFrame(params_tab).to_csv(C.TABELAS / "hiperparametros.csv", index=False)
    melhor = res.loc[0, "modelo"]
    print(f"\nMelhor modelo (PR-AUC out-of-fold no treino): {melhor}")
    cols = ["modelo", "familia", "pr_auc_oof", "roc_auc", "pr_auc", "precisao", "recall", "f1",
            "recall_top10%", "lift_top10%"]
    print(res[cols].round(3).to_string(index=False))

    cores = dict(zip(res["modelo"], sns.color_palette("tab10", len(res))))

    fig, ax = plt.subplots(figsize=(7, 6))
    for nome in res["modelo"]:
        fpr, tpr, _ = roc_curve(yte, scores_te[nome])
        ax.plot(fpr, tpr, color=cores[nome], label=f"{nome} ({roc_auc_score(yte, scores_te[nome]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos",
           title=f"Curvas ROC - teste ({C.OFERTA_TESTE})")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.FIG_MODELOS / "01_roc.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for nome in res["modelo"]:
        p, r, _ = precision_recall_curve(yte, scores_te[nome])
        ax.plot(r, p, color=cores[nome], label=f"{nome} ({average_precision_score(yte, scores_te[nome]):.3f})")
    ax.axhline(yte.mean(), color="k", ls="--", lw=1, label=f"acaso ({yte.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precisão", title=f"Curvas Precisão-Recall - teste ({C.OFERTA_TESTE})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIG_MODELOS / "02_precisao_recall.png", dpi=130)
    plt.close(fig)

    m = res.melt(id_vars="modelo", value_vars=["roc_auc", "pr_auc", "f1", "recall"],
                 var_name="métrica", value_name="valor")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=m, x="modelo", y="valor", hue="métrica", ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set(title="Comparação de modelos no teste", xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(C.FIG_MODELOS / "03_comparacao_metricas.png", dpi=130)
    plt.close(fig)

    n = len(res)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(4 * int(np.ceil(n / 2)), 7))
    for ax, nome in zip(axes.ravel(), res["modelo"]):
        lim = res.set_index("modelo").loc[nome, "limiar"]
        cm = confusion_matrix(yte, (scores_te[nome] >= lim).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["permanece", "evade"], yticklabels=["permanece", "evade"])
        ax.set_title(nome, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Matrizes de confusão no teste (limiar F1 escolhido no treino)")
    fig.tight_layout()
    fig.savefig(C.FIG_MODELOS / "04_matrizes_confusao.png", dpi=130)
    plt.close(fig)

    arv = ajustados["Árvore de Decisão"]
    nomes = arv.named_steps["pre"].get_feature_names_out()
    nomes = [n.split("__", 1)[1] for n in nomes]
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(arv.named_steps["clf"], max_depth=3, feature_names=nomes, class_names=["permanece", "evade"],
              filled=True, rounded=True, fontsize=8, ax=ax, proportion=True, impurity=False)
    ax.set_title("Árvore de decisão (3 primeiros níveis)")
    fig.tight_layout()
    fig.savefig(C.FIG_MODELOS / "05_arvore_decisao.png", dpi=110)
    plt.close(fig)
    regras = export_text(arv.named_steps["clf"], feature_names=nomes, max_depth=4)
    (C.TABELAS / "arvore_regras.txt").write_text(regras, encoding="utf-8")
    imp_arv = pd.Series(arv.named_steps["clf"].feature_importances_, index=nomes).sort_values(ascending=False)
    imp_arv.head(20).round(4).to_csv(C.TABELAS / "importancia_arvore.csv", header=["importancia"])

    print(f"\nImportância por permutação para '{melhor}'...")
    amostra = Xte.sample(min(2500, len(Xte)), random_state=C.SEED)
    ya = yte[Xte.index.get_indexer(amostra.index)]
    pi = permutation_importance(ajustados[melhor], amostra, ya, scoring="average_precision",
                                n_repeats=3, random_state=C.SEED, n_jobs=1)
    imp = pd.DataFrame({"variavel": feats, "importancia": pi.importances_mean, "desvio": pi.importances_std})
    imp = imp.sort_values("importancia", ascending=False)
    imp.round(5).to_csv(C.TABELAS / "importancia_permutacao.csv", index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["variavel"], top["importancia"], xerr=top["desvio"], color="#4C78A8")
    ax.set(title=f"Importância por permutação (queda de PR-AUC) - {melhor}")
    fig.tight_layout()
    fig.savefig(C.FIG_MODELOS / "06_importancia_permutacao.png", dpi=130)
    plt.close(fig)
    print(imp.head(10).round(4).to_string(index=False))

    s = scores_te[melhor]
    lim = res.set_index("modelo").loc[melhor, "limiar"]
    pr = te[[C.COL_ID, "codigo_modulo", C.COL_OFERTA, C.ALVO]].copy()
    pr["score_risco"] = s
    pr["previsao_evasao"] = (s >= lim).astype(int)
    pr["percentil_risco"] = pd.Series(s).rank(pct=True).values
    pr["faixa_risco"] = pd.cut(pr["percentil_risco"], [0, 0.7, 0.9, 1.0],
                               labels=["Baixo", "Médio", "Alto"], include_lowest=True)
    pr.sort_values("score_risco", ascending=False).to_csv(C.TABELAS / "predicoes_teste.csv", index=False)
    faixas = pr.groupby("faixa_risco", observed=True)[C.ALVO].agg(n="count", evasao_real="mean").round(3)
    faixas.to_csv(C.TABELAS / "evasao_por_faixa_risco.csv")
    print(f"\nEvasão real por faixa de risco (teste):\n{faixas.to_string()}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(faixas.index.astype(str), faixas["evasao_real"] * 100, color=["#4C78A8", "#F58518", "#E45756"])
    ax.axhline(yte.mean() * 100, color="k", ls="--", lw=1, label="média geral")
    ax.set(ylabel="Evasão real (%)", title=f"Faixas de risco - {melhor}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_MODELOS / "07_faixas_risco.png", dpi=130)
    plt.close(fig)

    joblib.dump({"modelo": ajustados[melhor], "limiar": lim, "features": feats, "nome": melhor},
                C.MODELOS / "melhor_modelo.joblib")
    print(f"\nConcluído em {(time.time() - t0) / 60:.1f} min. Modelo salvo em {C.MODELOS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="grades e árvores reduzidas (teste rápido)")
    main(ap.parse_args().rapido)