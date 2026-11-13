from eda import analisarDados
from preprocessamento import carregarDados
from avaliacao import (escolherLimiar, avaliarModeloCompleto, resumirModelos, faixasDeRisco, pontuacao,
                       importanciaPermutacao)
from graficos import (gerarGraficosDados, graficoClusters, graficoROC, graficoPrecisaoRecall,
                      graficoMatrizConfusao, graficoImportancia, graficoCortes)
from modelos import PARAMETROS_PADRAO, criarTodosModelos
from regressao import compararC, mostrarCoeficientes
from arvore import compararArvore, mostrarRegras, mostrarImportancias
from ensembles import compararEnsembles
from svm import compararSVM
from kmeans import compararKMeans, treinarKMeans, perfilClusters
from selecao import compararSelecao
from cortes import compararCortes
from previsao import preverNovoAluno
from alunos import ALUNOS

# ______________________
# CONFIGURAÇÃO:
analisarExploratoria = False    # imprime tabelas da análise exploratória
avaliar = True
analisarRegressao = False
analisarArvore = False
analisarEnsembles = False
analisarSVM = False
analisarClusters = True
selecionar = False
analisarImportancia = False     # importância por permutação do melhor modelo (~1 min)
analisarCortes = False          # demora ~12 min (6 cortes x 6 modelos); precisa de data/cortes/ (ver setup)
prever = True
gerarGraficos = False           # salva as figuras do relatório em reports/figures/ (liga a importância)
escolhidos = None               # None = todos os modelos; ex.: ["Random Forest", "SVM RBF"] roda só esses
limiares = (0.5, 0.3)           # limiares fixos mostrados além do limiar F1 escolhido no treino
# ______________________

if analisarExploratoria:
    analisarDados("data/oulad_evasao.csv")

X_train, X_test, y_train, y_test, grupos = carregarDados("data/oulad_evasao.csv")
print(f"Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas | Colunas: {X_train.shape[1]}")

# Valores padrão; se a análise estiver ligada, são substituídos pelos escolhidos na validação (só no treino)
p = {chave: dict(valor) for chave, valor in PARAMETROS_PADRAO.items()}

if analisarRegressao:
    p["regressao"]["c"] = compararC(X_train, y_train, grupos, (0.01, 0.1, 1))

if analisarArvore:
    p["arvore"]["profundidade"], p["arvore"]["folha"] = compararArvore(X_train, y_train, grupos)

if analisarEnsembles:
    p["random_forest"], p["extra_trees"], p["gradient_boosting"] = compararEnsembles(X_train, y_train, grupos)

if analisarSVM:
    p["svm_linear"], p["svm_rbf"] = compararSVM(X_train, y_train, grupos)

if analisarClusters:
    k = compararKMeans(X_train)
    kmeans = treinarKMeans(X_train, k)
    perfil = perfilClusters(kmeans, X_train, y_train, X_test, y_test)

if selecionar:
    compararSelecao(X_train, y_train, grupos, p["regressao"]["c"])

modelos = criarTodosModelos(p)
if escolhidos is not None:
    modelos = {nome: modelo for nome, modelo in modelos.items() if nome in escolhidos}

resultados, pr_validacao, limiar_f1, escores = {}, {}, {}, {}
if avaliar or prever:
    for nome, modelo in modelos.items():
        print(f"\n>>> Treinando {nome}...")
        limiar_f1[nome], pr_validacao[nome] = escolherLimiar(modelo, X_train, y_train, grupos)
        modelo.fit(X_train, y_train)
        escores[nome] = pontuacao(modelo, X_test)
        if avaliar:
            resultados[nome] = avaliarModeloCompleto(nome, modelo, X_test, y_test, limiar_f1[nome], limiares)

if avaliar:
    resumo = resumirModelos(resultados, pr_validacao)
    melhor = resumo.index[0]
    faixasDeRisco(escores[melhor], y_test)

    if analisarImportancia or gerarGraficos:
        importancias = importanciaPermutacao(modelos[melhor], X_test, y_test)

if analisarRegressao:
    mostrarCoeficientes(modelos["Regressão Logística (baseline)"])

if "Árvore de Decisão" in modelos and (avaliar or analisarArvore):
    mostrarRegras(modelos["Árvore de Decisão"], profundidade=3)
    mostrarImportancias(modelos["Árvore de Decisão"])

if prever:
    nome_prever = melhor if avaliar else list(modelos)[0]
    print(f"\n########## Previsão para novos alunos com o modelo: {nome_prever} ##########")
    for nome, aluno in ALUNOS.items():
        print(f"\n########## Aluno: {nome} ##########")
        preverNovoAluno(modelos[nome_prever], aluno, limiar_f1[nome_prever], X_train)

if analisarCortes:
    resultados_cortes = compararCortes()
    if gerarGraficos:
        graficoCortes(resultados_cortes)

if gerarGraficos and avaliar:
    print("\n########## Gerando figuras ##########")
    gerarGraficosDados("data/oulad_evasao.csv")
    if analisarClusters:
        graficoClusters(perfil, y_test.mean())
    graficoROC(escores, y_test, resumo.index)
    graficoPrecisaoRecall(escores, y_test, resumo.index)
    graficoMatrizConfusao(melhor, escores[melhor], y_test, limiar_f1[melhor])
    graficoImportancia(importancias, melhor)
