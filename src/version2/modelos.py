from collections import OrderedDict

from arvore import criarModeloArvore
from ensembles import criarExtraTrees, criarGradientBoosting, criarRandomForest, criarVoting
from regressao import criarModelo as criarRegressao
from svm import criarSVMLinear, criarSVMRBF

# Parâmetros padrão: são os escolhidos pela validação cruzada (analisarArvore/analisarEnsembles/analisarSVM
# no main.py refazem essa busca). Ficam aqui para o main e o experimento de cortes usarem os mesmos valores.
PARAMETROS_PADRAO = {
    "regressao": {"c": 1},
    "arvore": {"profundidade": 10, "folha": 100},
    "random_forest": {"profundidade": 12, "folha": 30},
    "extra_trees": {"profundidade": 14, "folha": 20},
    "gradient_boosting": {"taxa": 0.03, "profundidade": 5},
    "svm_linear": {"c": 1},
    "svm_rbf": {"c": 0.5, "gamma": 0.01},
}

NOMES = ["Regressão Logística (baseline)", "Árvore de Decisão", "Random Forest", "Extra Trees",
         "Gradient Boosting", "SVM Linear", "SVM RBF", "Voting (RF + GB + RL)"]


def criarTodosModelos(p=None):
    p = PARAMETROS_PADRAO if p is None else p
    regressao = criarRegressao(**p["regressao"])
    random_forest = criarRandomForest(**p["random_forest"])
    gradient_boosting = criarGradientBoosting(**p["gradient_boosting"])

    return OrderedDict([
        (NOMES[0], regressao),
        (NOMES[1], criarModeloArvore(**p["arvore"])),
        (NOMES[2], random_forest),
        (NOMES[3], criarExtraTrees(**p["extra_trees"])),
        (NOMES[4], gradient_boosting),
        (NOMES[5], criarSVMLinear(**p["svm_linear"])),
        (NOMES[6], criarSVMRBF(**p["svm_rbf"])),
        (NOMES[7], criarVoting(random_forest, gradient_boosting, regressao)),
    ])
