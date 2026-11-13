from sklearn.model_selection import cross_val_score

from avaliacao import cv
from ensembles import criarGradientBoosting
from regressao import criarModelo

# Grupos de colunas testados. Cada grupo é REMOVIDO do treino e comparado com o modelo completo.
# (os grupos são definidos a partir dos nomes das colunas, então servem para qualquer corte)
TOLERANCIA = 0.005   # se a PR-AUC cair menos que isso ao remover o grupo, ele é considerado descartável


def gruposPadrao(X_train):
    colunas = list(X_train.columns)
    tipos = [c for c in colunas if c.startswith("log_vle_cliques_")
             and c not in ("log_vle_cliques_pre_curso", "log_vle_cliques_ult_14d", "log_vle_cliques_ult_30d")]
    return {
        "todas (referencia)": [],
        "sem demografia": ["genero_masc", "deficiencia_sim", "escolaridade_ord", "imd_ord", "faixa_etaria_ord",
                           "faixa_imd_ausente", "regiao"],
        "sem tipos de atividade": tipos,
        "sem avaliacoes": [c for c in colunas if c.startswith("aval_")],
        "sem VLE (tudo)": [c for c in colunas if c.startswith(("vle_", "log_vle_")) or c == "nunca_acessou_vle"],
    }


def _prAuc(modelo, X, y, grupos):
    # Os mesmos folds (cv) em todas as chamadas: a diferença entre grupos vem da coluna removida,
    # e não de uma divisão diferente dos dados.
    pr = cross_val_score(modelo, X, y, groups=grupos, cv=cv, scoring="average_precision")
    return pr.mean()


def compararSelecao(X_train, y_train, grupos, c=1, grupos_removidos=None):
    grupos_removidos = gruposPadrao(X_train) if grupos_removidos is None else grupos_removidos

    print("\n------------- Selecao de atributos por validacao cruzada (somente treino) -------------")
    print(f"{len(X_train)} linhas | Regressao Logistica (C={c}) e Gradient Boosting")
    print(f"Descartavel se a PR-AUC cair menos que {TOLERANCIA}\n")
    print(f"{'grupo removido':<26}{'colunas':>8}{'LR PR':>9}{'dif':>9}{'GB PR':>9}{'dif':>9}")

    resultados = {}
    ref_lr = ref_gb = None
    for nome, colunas in grupos_removidos.items():
        validas = [col for col in colunas if col in X_train.columns]
        Xt = X_train.drop(columns=validas)

        pr_lr = _prAuc(criarModelo(c), Xt, y_train, grupos)
        pr_gb = _prAuc(criarGradientBoosting(iteracoes=100), Xt, y_train, grupos)

        # o primeiro grupo (lista vazia) é a referência
        if ref_lr is None:
            ref_lr, ref_gb = pr_lr, pr_gb

        print(f"{nome:<26}{len(validas):>8}{pr_lr:>9.4f}{pr_lr - ref_lr:>+9.4f}{pr_gb:>9.4f}{pr_gb - ref_gb:>+9.4f}")
        resultados[nome] = {"lr": pr_lr, "gb": pr_gb, "colunas": validas}

    return resultados
