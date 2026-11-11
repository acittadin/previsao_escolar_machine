# Educação e Impacto Social: Previsão de Evasão Escolar

Projeto de Machine Learning sobre o **Open University Learning Analytics Dataset (OULAD)**.
Objetivo: identificar, **no dia 90 do curso**, estudantes com maior risco de evadir (`Withdrawn`), para que a
instituição possa intervir a tempo.

Técnicas aplicadas: **limpeza e engenharia de dados, análise exploratória, KMeans, Árvore de Decisão,
Ensembles (Random Forest, Extra Trees, Gradient Boosting, Voting) e SVM (linear e RBF)**.

---

## Estrutura

```
projeto_evasao_oulad/
├── main.py                     # executa o pipeline (python main.py)
├── requirements.txt
├── data/
│   ├── raw/                    # coloque aqui os 7 CSVs do OULAD (não incluídos: ~450 MB)
│   └── processed/
│       ├── cortes/                 # datasets agregados para cortes de 0, 30, 60, 90, 120 e 180 dias
│       ├── dataset_evasao_pt.csv   # agregado + traduzido (entrada da limpeza)
│       ├── dataset_limpo.csv       # saída da limpeza
│       └── dicionario_dados.csv
├── src/
│   ├── config.py               # caminhos, semente, corte de 90 dias, oferta de teste
│   ├── agregacao.py            # CSVs brutos -> 1 linha por estudante x módulo x oferta
│   ├── traducao.py             # colunas e categorias em português
│   ├── limpeza.py              # limpeza determinística + split treino/teste
│   ├── eda.py                  # análise exploratória
│   ├── clusterizacao.py        # KMeans
│   ├── modelagem.py            # árvore, ensembles, SVM
│   └── experimento_cortes.py   # experimento com diferentes dias de corte
├── reports/
│   ├── figures/{eda,clusters,modelos}/
│   └── tables/                 # todas as métricas e tabelas em CSV
└── models/                     # kmeans.joblib e melhor_modelo.joblib
```

## Como executar

```bash
pip install -r requirements.txt
python main.py                  # limpeza -> EDA -> KMeans -> modelagem (~5 min, 1 núcleo)
python main.py --agregar        # inclui a agregação (exige os CSVs em data/raw, alguns minutos)
python main.py --etapas eda kmeans
python main.py --rapido         # modelagem com grades reduzidas (~2 min)
python main.py --etapas cortes  # experimento com vários dias de corte (~12 min)
```

O dataset já agregado está em `data/processed/`, então o projeto roda sem os CSVs brutos.
Os CSVs originais estão em https://analyse.kmi.open.ac.uk/open_dataset (ou no Kaggle, como `archive.zip`).

---

## Decisões metodológicas

| Decisão | Motivo |
|---|---|
| **Unidade de análise**: estudante × módulo × oferta | É a granularidade em que a evasão acontece |
| **Alvo**: `evasao = 1` se `final_result == Withdrawn` | Definição direta de evasão |
| **Corte no dia 90** | Só usa informação disponível até lá (previsão precoce, sem vazamento temporal) |
| **Remoção de quem já cancelou até o dia 90** (7.031) | Para eles a evasão já é conhecida; não há o que prever |
| **`date_unregistration` e `resultado_final` fora das features** | Vazam o alvo |
| **Split temporal**: treino 2013B, 2013J, 2014B; teste 2014J | Simula o uso real: treinar com o passado e prever a oferta seguinte |
| **Validação cruzada agrupada por estudante** (GroupKFold) | O mesmo estudante aparece em várias ofertas; evita vazar entre folds |
| **Imputação, escala e one-hot dentro do `Pipeline`** | Ajustados só no treino |
| **`class_weight="balanced"`** | Evasão é ~12% (classes desbalanceadas) |
| **Métrica principal: PR-AUC**, mais F1/recall da classe "evade" | Acurácia é enganosa com classes desbalanceadas |
| **Limiar de decisão escolhido por F1 nas previsões out-of-fold do treino** | O teste nunca é usado para escolher nada |
| **Melhor modelo escolhido pela PR-AUC out-of-fold**, não pelo teste | Evita seleção contaminada pelo teste |

**Limpeza (`src/limpeza.py`)**: remoção de colunas que vazam o alvo ou são redundantes (`vle_ultimo_acesso`);
flags de ausência (`faixa_imd_ausente`, `aval_sem_nota`); quem nunca acessou o VLE recebe 0 cliques; codificação ordinal
de escolaridade, IMD e idade; `log1p` nas colunas de cliques (assimetria ~3); demais ausências imputadas pela mediana no pipeline.

## Resultados

### Análise exploratória (`reports/figures/eda/`)
- Evasão geral de **12,3%** (3.133 de 25.562 registros).
- Os sinais mais fortes são de **desempenho e entrega**: nota média (-0,17), taxa de entrega (-0,16) e atraso médio (+0,13)
  (correlação de Spearman com a evasão). Quem evade tem nota média de 67 contra 76 de quem permanece.
- **Cliques recentes pesam mais que o total**: mediana de 4 cliques nos últimos 14 dias para quem evade, contra 19.
- Fatores demográficos têm efeito modesto (maior evasão com deficiência declarada, tentativas anteriores e menor IMD).
- A evasão varia bastante por módulo (de 5,6% em GGG a 19,5% em CCC).

### KMeans (`reports/figures/clusters/`)
Ajustado só no treino, sem usar o alvo; k = 6 (melhor silhueta entre 3 e 6).
**A silhueta é baixa (0,20): os grupos são pouco separados**, então os clusters servem como perfis descritivos, não como
segmentos nítidos. Mesmo assim, o cluster de estudantes **desengajados** (~67 dias sem acesso, ~14% de entregas)
teve **28,5% de evasão no teste**, contra 12,9% na média; o cluster mais engajado teve 5,5%.

### Modelos supervisionados (teste = oferta 2014J; prevalência de evasão = 12,9%)

| Modelo | PR-AUC (OOF treino) | ROC-AUC | PR-AUC | Precisão | Recall | F1 | Recall no top 10% |
|---|---|---|---|---|---|---|---|
| **Voting (RF + GB + LR)** *(melhor no treino)* | 0,294 | 0,707 | 0,262 | 0,246 | 0,466 | 0,322 | 21,7% |
| Gradient Boosting | 0,290 | 0,702 | 0,251 | 0,241 | 0,488 | 0,323 | 20,7% |
| SVM RBF | 0,285 | 0,719 | 0,286 | 0,276 | 0,438 | 0,338 | 27,3% |
| Extra Trees | 0,283 | 0,723 | **0,301** | 0,288 | 0,493 | **0,364** | **27,8%** |
| Random Forest | 0,282 | 0,715 | 0,281 | 0,247 | 0,508 | 0,333 | 25,7% |
| SVM Linear | 0,282 | 0,680 | 0,228 | 0,222 | 0,464 | 0,300 | 19,8% |
| Regressão Logística (baseline) | 0,282 | 0,679 | 0,224 | 0,225 | 0,430 | 0,296 | 19,3% |
| Árvore de Decisão | 0,249 | 0,652 | 0,191 | 0,213 | 0,549 | 0,307 | 15,8% |

Leituras principais:
- **Os modelos superam o acaso** (PR-AUC 0,19 a 0,30 contra 0,129), mas o desempenho é **moderado**: com a informação
  disponível até o dia 90, a evasão é difícil de prever com precisão.
- **Ensembles e SVM RBF superam a árvore única e os modelos lineares**, o que sugere relações não lineares.
  A árvore de decisão é a mais interpretável (`reports/figures/modelos/05_arvore_decisao.png`, `reports/tables/arvore_regras.txt`),
  porém a menos precisa.
- As diferenças entre os quatro melhores no teste (Extra Trees, SVM RBF, Random Forest, Voting) são pequenas e
  **não devem ser tratadas como definitivas**: há um único conjunto de teste e nenhuma repetição com outras sementes.
  O Voting é o "melhor" por ter sido escolhido pelo critério definido antes do teste (PR-AUC out-of-fold).
- **Uso prático**: ordenando os estudantes do teste por risco, a evasão real foi de **8,3%** na faixa "baixo" (70% dos estudantes),
  **21,4%** em "médio" e **27,9%** em "alto" (10% mais arriscados), ou seja, cerca de 2,2 vezes a taxa média.
  Lista completa em `reports/tables/predicoes_teste.csv`.
- **Variáveis mais importantes** (permutação): nota ponderada acumulada, taxa de entrega, módulo, cliques nos últimos
  14 dias e atraso médio nas entregas (`reports/tables/importancia_permutacao.csv`).

Hiperparâmetros escolhidos em `reports/tables/hiperparametros.csv`. Dois deles ficaram na borda da grade
(`C=1` no SVM linear e `C=0,5` no SVM RBF), então uma busca mais ampla poderia render pequenos ganhos.

## Experimento: e se a previsão for feita em outro momento do curso?

Repeti o pipeline com cortes de **0, 30, 60, 90, 120 e 180 dias** (`src/experimento_cortes.py`). Em cada corte, só entram
eventos até aquele dia e saem da base os estudantes que já cancelaram até ele. Os modelos e os hiperparâmetros são os mesmos
(escolhidos no corte de 90 dias), o split é o mesmo (teste = 2014J) e o limiar vem do treino. Resultados em
`reports/tables/cortes_*.csv` e `reports/figures/cortes/`.

| Corte (dias) | Estudantes ativos (teste) | Evasão ainda por acontecer (teste) | Extra Trees: ROC-AUC | Extra Trees: PR-AUC ÷ acaso | Extra Trees: lift top 10% |
|---|---|---|---|---|---|
| 0 | 10.363 | 28,3% | 0,682 | 1,61x | 1,94 |
| 30 | 9.203 | 19,2% | 0,698 | 2,02x | 2,56 |
| 60 | 8.818 | 15,7% | 0,710 | 2,21x | 2,72 |
| 90 | 8.533 | 12,9% | 0,723 | 2,33x | 2,78 |
| 120 | 8.260 | 10,0% | 0,741 | 2,50x | 2,92 |
| 180 | 7.719 | 3,7% | 0,771 | 2,54x | 2,98 |

**Como ler (cuidado com a comparação entre cortes):**
- **PR-AUC e F1 absolutos caem muito com o corte** (por exemplo, F1 de ~0,50 no dia 0 para ~0,14 no dia 180), mas isso se deve
  principalmente à queda da evasão restante: de 28,3% para 3,7%. Com menos casos positivos, essas métricas pioram mesmo que o
  modelo discrimine melhor. **Elas não são comparáveis entre cortes.**
- As métricas comparáveis são **ROC-AUC, PR-AUC ÷ prevalência e lift**, e as três **melhoram com o corte**. Cada dia a mais de
  histórico dá ao modelo mais sinal (notas, entregas, cliques recentes) sobre quem ainda vai sair.
- **Já no dia 0** (só com dados demográficos, de matrícula e módulo, mais cliques pré-curso) os modelos têm ROC-AUC de cerca de
  0,68 a 0,70 (a árvore única fica em 0,64), acima do acaso, mas a melhora até o dia 90 (0,72) é modesta.
- **O ganho tem um custo operacional.** No dia 180 restam só cerca de 290 evasões no teste, com precisão de ~8% no limiar de F1.
  Mesmo com ROC-AUC mais alto (0,77), sobra pouco tempo para intervir e quase tudo o que o modelo sinaliza é falso positivo.
  Já no dia 0, ele sinaliza muita gente (evasão de 28%), mas com pouca capacidade de separar quem realmente vai evadir.
- **O melhor modelo varia com o corte** (critério: PR-AUC out-of-fold no treino): Regressão Logística nos cortes 0 e 30, SVM RBF
  no 60 e Gradient Boosting no 90, 120 e 180. As curvas por modelo são irregulares (por exemplo, o lift do Gradient Boosting
  salta no corte 120 e cai no 180), o que indica **ruído de um único split e uma única semente**. A conclusão robusta é a tendência
  geral, não a posição de cada modelo em cada corte.
- Um ponto intermediário (30 a 90 dias) parece um bom compromisso entre tempo para intervir e capacidade de previsão, mas essa é
  uma leitura dos dados, não uma recomendação: a escolha depende do custo da intervenção e de quanto tempo ela leva para fazer efeito.

## Limitações

- **Experimento de cortes:** os hiperparâmetros foram ajustados só no corte de 90 dias e reaproveitados nos demais, o que pode
  favorecê-lo levemente; e a população muda a cada corte (saem os que já cancelaram), então os cortes avaliam problemas diferentes.
- **Um único split temporal** (teste = 2014J): sem intervalos de confiança. Vale repetir com outros cortes
  (por exemplo, testar em 2014B) e com várias sementes para medir a variabilidade.
- **Estudantes repetidos**: parte dos estudantes do teste também aparece no treino em ofertas anteriores
  (histórico legítimo em uso real), mas o desempenho pode diferir para estudantes nunca vistos.
- **Prever a evasão não significa explicá-la**: as importâncias mostram associação, não causa.
- **Equidade**: variáveis como deficiência, idade e região entram no modelo; antes de qualquer uso real, é preciso
  auditar se o desempenho é parecido entre grupos e se o uso do modelo não prejudica ninguém.
- O dataset é de uma universidade a distância do Reino Unido; os resultados podem não se transferir para outros contextos.

## Extensões sugeridas
1. Reajustar os hiperparâmetros em cada corte e repetir com várias sementes e outros splits temporais, para estimar a variabilidade.
2. Usar o cluster do KMeans como feature adicional dos modelos supervisionados.
3. Calibrar as probabilidades e analisar custo-benefício de intervenção por faixa de risco.
4. Auditoria de equidade (desempenho por gênero, deficiência, faixa etária e região).
5. Explicabilidade local (por exemplo, SHAP) para justificar o risco de cada estudante.
