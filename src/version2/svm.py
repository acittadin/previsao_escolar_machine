from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from avaliacao import compararParametros
from preprocessamento import criarPreparo

# O SVM usa distância até uma fronteira de separação: SEM padronizar, colunas com escala grande
# (como créditos ou dias) dominariam as outras. Por isso escalar=True.


def criarSVMLinear(c=1):
    # dual=False: formulação primal, bem mais rápida que a padrão quando há mais linhas que colunas
    return Pipeline(steps=[
        ("prep", criarPreparo(escalar=True)),
        ("svm", LinearSVC(C=c, class_weight="balanced", dual=False, max_iter=5000, random_state=42))
    ])


def criarSVMRBF(c=0.5, gamma=0.01):
    # kernel RBF: fronteira NÃO linear. C = quanto o modelo tolera erros no treino;
    # gamma = alcance da influência de cada exemplo (gamma alto = fronteira mais "recortada")
    return Pipeline(steps=[
        ("prep", criarPreparo(escalar=True)),
        ("svm", SVC(kernel="rbf", C=c, gamma=gamma, class_weight="balanced", cache_size=1000, random_state=42))
    ])


def compararSVM(X_train, y_train, grupos, tamanho_amostra=6000):
    # O SVC cresce ~ com o quadrado do número de linhas, então a busca do RBF usa uma AMOSTRA do treino
    # (o teste não é tocado) e o modelo final é reajustado no treino completo.
    linear = compararParametros(criarSVMLinear, {"c": (0.01, 0.1, 1)}, X_train, y_train, grupos)
    rbf = compararParametros(criarSVMRBF, {"c": (0.5, 2), "gamma": ("scale", 0.01)}, X_train, y_train, grupos,
                             tamanho_amostra=tamanho_amostra)
    return linear, rbf
