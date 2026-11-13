# Previsão de evasão escolar (OULAD)

Projeto de Machine Learning que prevê, **no 90º dia do curso**, quais estudantes de educação a distância têm maior risco de evadir
(cancelar a matrícula), usando o *Open University Learning Analytics Dataset* (OULAD).

Técnicas: **K-Means**, **Árvore de Decisão**, **ensembles** (Random Forest, Extra Trees, Gradient Boosting, Voting) e **SVM** (linear e RBF),
com **Regressão Logística** como modelo de referência. Mesma organização do repositório `machine-learning-satc`
(`setup/` + `src/version1` + `src/version2`, um módulo por assunto, `main.py` dirigido por flags).

## Estrutura

```
previsao_escolar_machine/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                      # coloque aqui os 7 CSVs originais do OULAD (não vão para o git)
│   ├── oulad_evasao.csv          # dataset agregado (corte de 90 dias), gerado pelo setup
│   └── cortes/                   # um dataset por corte (0, 30, 60, 90, 120, 180 dias)
├── setup/
│   └── agregarOulad.py           # roda UMA vez: CSVs originais -> data/oulad_evasao.csv
├── reports/
│   ├── figures/                  # figuras do relatório (geradas com gerarGraficos = True)
│   └── tables/                   # cortes_resultados.csv (gerado pelo experimento de cortes)
└── src/
    ├── version1/                 # primeira abordagem (simples)
    │   ├── main.py  preprocessamento.py  arvore.py  ensembles.py  svm.py  avaliacao.py
    └── version2/                 # versão final
        ├── main.py               # flags de configuração + fluxo completo
        ├── preprocessamento.py   # limpeza, tratamento e divisão treino/teste (temporal)
        ├── eda.py                # análise exploratória (tabelas)
        ├── regressao.py  arvore.py  ensembles.py  svm.py    # um módulo por família de modelo
        ├── modelos.py            # parâmetros padrão e criação dos 8 modelos
        ├── kmeans.py             # agrupamento (perfis de estudantes)
        ├── graficos.py           # figuras do relatório (salvas em reports/figures/)
        ├── avaliacao.py          # validação, limiar, métricas, lift, faixas de risco
        ├── selecao.py            # seleção de atributos por grupos
        ├── previsao.py  alunos.py   # previsão para novos alunos (alunos de exemplo)
        └── cortes.py             # experimento: e se a previsão for feita em outro dia do curso?
```

## Como executar

1. Instale as dependências (testado com Python 3.12):

```bash
pip install -r requirements.txt
```

   (pandas, numpy, scikit-learn >= 1.2, matplotlib e seaborn; as duas últimas só são usadas para gerar as figuras)

2. Baixe o OULAD (`archive.zip`, em https://analyse.kmi.open.ac.uk/open_dataset, licença CC BY 4.0) e extraia os 7 CSVs em `data/raw/`.

3. Gere o dataset agregado (apenas uma vez), a partir da **raiz do projeto**:

```bash
python setup/agregarOulad.py                              # corte de 90 dias -> data/oulad_evasao.csv
python setup/agregarOulad.py --cortes 0 30 60 90 120 180  # também gera data/cortes/ (para o experimento de cortes)
```

4. Rode uma das versões, também a partir da raiz:

```bash
python src/version1/main.py     # ~1 min
python src/version2/main.py     # ~3 min (8 modelos, 1 núcleo)
```

## Flags do `src/version2/main.py`

| Flag | O que faz |
|---|---|
| `analisarExploratoria` | imprime as tabelas da análise exploratória |
| `avaliar` | treina os 8 modelos, escolhe o limiar no treino e avalia no teste |
| `analisarRegressao`, `analisarArvore`, `analisarEnsembles`, `analisarSVM` | refazem a busca de parâmetros por validação cruzada (só no treino) |
| `analisarClusters` | compara k do KMeans e imprime o perfil dos clusters |
| `selecionar` | testa remover grupos de atributos (demografia, tipos de atividade, avaliações, VLE) |
| `analisarImportancia` | importância por permutação do melhor modelo (queda da PR-AUC no teste ao embaralhar cada variável) |
| `analisarCortes` | experimento com cortes de 0 a 180 dias (~10 min; precisa de `data/cortes/`; salva `reports/tables/cortes_resultados.csv`) |
| `prever` | estima o risco de 3 alunos de exemplo (`alunos.py`) |
| `gerarGraficos` | salva as 9 figuras do relatório em `reports/figures/` (liga também a importância por permutação; a figura de cortes exige `analisarCortes`) |
| `escolhidos` | restringe os modelos (ex.: `["Random Forest", "SVM RBF"]`) |
| `limiares` | limiares fixos mostrados além do limiar F1 escolhido no treino |

## Do dado ao modelo

- **Unidade**: 1 linha = 1 estudante em 1 oferta de módulo. 25.562 linhas; **12,3% evadem** (classe desbalanceada).
- **Corte de 90 dias**: só entram eventos (cliques, notas, entregas) até o dia 90; quem já cancelou até lá (7.031) é removido.
- **Vazamento evitado**: `resultado_final` e `date_unregistration` revelam o desfecho e ficam fora.
- **Divisão temporal**: treino = ofertas 2013B, 2013J, 2014B (17.029 linhas); teste = 2014J (8.533 linhas). Na validação cruzada, o mesmo estudante fica sempre no mesmo fold (`GroupKFold`).
- **Pré-processamento dentro do pipeline** (imputação pela mediana, padronização, One-Hot): aprendido só no treino.
- **Desbalanceamento**: `class_weight="balanced"`. Métrica de validação: PR-AUC. O limiar de decisão (F1) e o "melhor modelo" saem só do treino; o teste nunca escolhe nada.

## `version1` x `version2`

| | version1 | version2 |
|---|---|---|
| Divisão | sorteio aleatório 70/30 | temporal (2014J no teste), com grupos por estudante |
| Nulos | `fillna(0)` | flags de ausência + valores determinísticos + mediana no pipeline |
| Desbalanceamento | ignorado | pesos de classe e limiar escolhido no treino |
| Métricas | acurácia, recall, ROC-AUC | PR-AUC, ROC-AUC, F1, recall, lift, faixas de risco |
| Modelos | Árvore, Random Forest, SVM | 8 modelos + KMeans + seleção de atributos + cortes |

Resultado da **version1** (teste aleatório, 7.669 linhas; baseline de acurácia = 87,7%):

| Modelo | Acurácia | Recall da evasão | ROC-AUC |
|---|---|---|---|
| Árvore de Decisão | 0,876 | 0,025 | 0,691 |
| Random Forest | 0,877 | 0,025 | 0,721 |
| SVM (RBF) | 0,877 | 0,000 | 0,651 |

A acurácia fica igual ao baseline e os modelos quase não encontram evasores: **a acurácia engana** nessa base. É o problema que a version2 resolve.

## Resultados da version2

Teste = oferta 2014J (8.533 estudantes; 12,9% evadem). Ordenado pela PR-AUC de validação no treino.

| Modelo | PR-AUC (treino) | Limiar | Precisão | Recall | F1 | ROC-AUC | PR-AUC | Lift top 10% |
|---|---|---|---|---|---|---|---|---|
| Voting (RF + GB + RL) | 0,294 | 0,574 | 0,246 | 0,466 | 0,322 | 0,707 | 0,262 | 2,17 |
| Gradient Boosting | 0,290 | 0,583 | 0,241 | 0,488 | 0,323 | 0,702 | 0,251 | 2,08 |
| SVM RBF | 0,285 | 0,423 | 0,276 | 0,438 | 0,338 | 0,719 | 0,286 | 2,73 |
| Extra Trees | 0,283 | 0,573 | 0,288 | 0,493 | 0,364 | 0,723 | 0,301 | 2,78 |
| Random Forest | 0,282 | 0,537 | 0,247 | 0,508 | 0,333 | 0,715 | 0,281 | 2,57 |
| SVM Linear | 0,282 | 0,139* | 0,222 | 0,464 | 0,300 | 0,680 | 0,228 | 1,98 |
| Regressão Logística (baseline) | 0,282 | 0,598 | 0,225 | 0,430 | 0,296 | 0,679 | 0,224 | 1,93 |
| Árvore de Decisão | 0,249 | 0,575 | 0,213 | 0,549 | 0,307 | 0,652 | 0,191 | 1,58 |

\* No SVM linear o limiar é aplicado à distância até a margem, não a uma probabilidade. Baseline da PR-AUC = prevalência = 0,129.

- Todos superam o acaso, mas o desempenho é **moderado**. As diferenças entre os quatro melhores são pequenas: há um único teste e uma única semente, então não devem ser tratadas como definitivas.
- **Faixas de risco** (Voting): evasão real de 8,3% no risco baixo (70% dos estudantes), 21,4% no médio e 27,9% nos 10% de maior risco (cerca de 2,2 vezes a média).
- **K-Means** (k = 6): silhueta de 0,20, ou seja, grupos pouco separados (perfis descritivos). O grupo dos desengajados (~67 dias sem acesso, 14% de entregas) teve 28,5% de evasão no teste.
- **Seleção de atributos** (PR-AUC de validação): remover as **avaliações** derruba cerca de 0,065 (são o grupo mais importante); remover a demografia custa 0,003 a 0,006; remover os 19 tipos de atividade do VLE muda a PR-AUC em menos de 0,004 (descartáveis).
- **Árvore**: a raiz divide pela nota ponderada acumulada (≤ 26,9); `mostrarRegras` imprime as regras.
- **Importância por permutação** (Voting, queda da PR-AUC no teste): nota ponderada acumulada (0,044), taxa de entrega (0,038), módulo (0,031) e cliques dos últimos 14 dias (0,024).
- O escore de `previsao.py` **não é uma probabilidade calibrada** (modelos com pesos de classe superestimam o risco): serve para ordenar estudantes e comparar com o limiar.

### Experimento de cortes (`analisarCortes`)

Mesmos modelos e parâmetros, mudando só o dia de corte (Extra Trees no teste):

| Corte (dias) | Estudantes no teste | Evasão restante | ROC-AUC | PR-AUC ÷ acaso | Lift top 10% | F1 |
|---|---|---|---|---|---|---|
| 0 | 10.363 | 28,3% | 0,682 | 1,61 | 1,94 | 0,499 |
| 30 | 9.203 | 19,2% | 0,698 | 2,02 | 2,56 | 0,414 |
| 60 | 8.818 | 15,7% | 0,710 | 2,21 | 2,72 | 0,375 |
| 90 | 8.533 | 12,9% | 0,723 | 2,33 | 2,78 | 0,364 |
| 120 | 8.260 | 10,0% | 0,741 | 2,50 | 2,92 | 0,322 |
| 180 | 7.719 | 3,7% | 0,771 | 2,54 | 2,98 | 0,145 |

ROC-AUC, PR-AUC ÷ prevalência e lift melhoram com o corte. **PR-AUC e F1 absolutos caem só porque sobra menos evasão para prever: não são comparáveis entre cortes.**

## Limitações

- Um único split temporal e uma única semente, sem intervalos de confiança.
- Parâmetros ajustados no corte de 90 dias e reaproveitados nos outros cortes.
- A população muda a cada corte (saem os que já cancelaram).
- Dados de uma universidade a distância do Reino Unido; pode não generalizar.
- Variáveis como deficiência, idade e região entram nos modelos: antes de qualquer uso real, auditar o desempenho por grupo (equidade).
- A importância dos atributos mostra associação, não causa.

## Reprodutibilidade

Semente fixa em 42. Todas as tabelas acima foram obtidas rodando `src/version1/main.py` e `src/version2/main.py` a partir dos CSVs originais
e coincidem com os resultados do projeto anterior (`projeto_evasao_oulad`) que serviu de base.
