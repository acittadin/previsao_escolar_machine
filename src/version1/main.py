from preprocessamento import carregarDados
from avaliacao import avaliarModelo
from arvore import treinarArvore, compararProfundidade
from ensembles import treinarRandomForest
from svm import treinarSVM

# ______________________
# CONFIGURAÇÃO:
avaliar = True
analisarArvore = False
# ______________________

X_train, X_test, y_train, y_test = carregarDados("data/oulad_evasao.csv")
print(f"Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas | Colunas: {X_train.shape[1]}")

profundidade = 5
if analisarArvore:
    profundidade = compararProfundidade(X_train, y_train)

modelo_arvore = treinarArvore(X_train, y_train, profundidade)
modelo_floresta = treinarRandomForest(X_train, y_train)
modelo_svm = treinarSVM(X_train, y_train)

if avaliar:
    avaliarModelo("Arvore de Decisao", modelo_arvore, X_test, y_test)
    avaliarModelo("Random Forest", modelo_floresta, X_test, y_test)
    avaliarModelo("SVM (RBF)", modelo_svm, X_test, y_test)
