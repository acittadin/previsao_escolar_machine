import pandas as pd

from avaliacao import calcularMetricas, escolherLimiar
from modelos import PARAMETROS_PADRAO, criarTodosModelos
from preprocessamento import carregarDados

# Experimento: como a capacidade de prever a evasão muda conforme o MOMENTO do curso em que a previsão é feita?
# Cada arquivo de data/cortes/ foi agregado com eventos só até aquele dia (setup/agregarOulad.py --cortes ...).
# Os parâmetros dos modelos são FIXOS (os do corte de 90 dias): assim a comparação isola o efeito da
# informação disponível. Limitação: os parâmetros podem favorecer um pouco o corte de 90.
# Obs.: no corte 0 ninguém tem avaliação vencida, então aval_taxa_entrega fica vazia e o imputer a ignora
# (aviso do scikit-learn, esperado e inofensivo).
CORTES_PADRAO = (0, 30, 60, 90, 120, 180)
MODELOS_CORTES = ["Regressão Logística (baseline)", "Árvore de Decisão", "Random Forest", "Extra Trees",
                  "Gradient Boosting", "SVM RBF"]


def compararCortes(cortes=CORTES_PADRAO, pasta="data/cortes", modelos_escolhidos=None):
    nomes = modelos_escolhidos or MODELOS_CORTES
    linhas = []
    for corte in cortes:
        print(f"\n################ CORTE DE {corte} DIAS ################")
        X_train, X_test, y_train, y_test, grupos = carregarDados(f"{pasta}/oulad_corte_{corte}.csv", corte)
        print(f"Treino: {len(X_train)} ({y_train.mean():.1%} evasao) | Teste: {len(X_test)} ({y_test.mean():.1%})")

        todos = criarTodosModelos(PARAMETROS_PADRAO)
        for nome in nomes:
            modelo = todos[nome]
            limiar, pr_treino = escolherLimiar(modelo, X_train, y_train, grupos)
            modelo.fit(X_train, y_train)
            metricas = calcularMetricas(modelo, X_test, y_test, limiar)
            linhas.append({"corte": corte, "modelo": nome, "n_teste": len(X_test), "pr_auc_treino": pr_treino,
                           **metricas})
            print(f"  {nome:<32} ROC={metricas['roc_auc']:.3f} PR={metricas['pr_auc']:.3f} "
                  f"F1={metricas['f1']:.3f} lift@10%={metricas['lift_top10']:.2f}")

    resultados = pd.DataFrame(linhas)
    resultados.round(4).to_csv("reports/tables/cortes_resultados.csv", index=False)
    print("\n" + "=" * 100)
    print("EVASAO AINDA POR ACONTECER (teste) e ROC-AUC por corte")
    print("=" * 100)
    print(resultados.groupby("corte")[["n_teste", "prevalencia"]].first().round(3).to_string())
    # PR-AUC e F1 ABSOLUTOS caem com o corte só porque sobra menos evasão para prever (prevalência menor):
    # não são comparáveis entre cortes. ROC-AUC, PR-AUC / prevalência e lift são comparáveis.
    for metrica, titulo in [("roc_auc", "ROC-AUC"), ("pr_auc_relativa", "PR-AUC / prevalencia (vezes o acaso)"),
                            ("lift_top10", "Lift entre os 10% de maior risco")]:
        print(f"\n{titulo}:")
        print(resultados.pivot(index="modelo", columns="corte", values=metrica).round(3).to_string())

    return resultados
