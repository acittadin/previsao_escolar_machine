# Previsão de evasão escolar (OULAD)

Projeto de Machine Learning que prevê, no 90º dia do curso, quais estudantes de educação a distância têm maior risco de evadir
(cancelar a matrícula), usando o *Open University Learning Analytics Dataset* (OULAD).

Técnicas: K-Means, árvore de Decisão, ensembles (Random Forest, Extra Trees, Gradient Boosting, Voting) e SVM (linear e RBF),
com Regressão Logística como modelo de referência.

1: Instalar as dependências:
```bash
pip install -r requirements.txt
```

2: Baixe o OULAD (`archive.zip`, em https://analyse.kmi.open.ac.uk/open_dataset, licença CC BY 4.0) e extraia os 7 CSVs em `data/raw/`.

3: Gere o dataset agregado (apenas uma vez):

```bash
python setup/agregarOulad.py                              # corte de 90 dias -> data/oulad_evasao.csv
python setup/agregarOulad.py --cortes 0 30 60 90 120 180  # também gera data/cortes/ (para o experimento de cortes)
```

4. Rode uma das versões, também a partir da raiz:

```bash
python src/version1/main.py     # Versão 1 feita, mas limitada
python src/version2/main.py     # Versão 2, demora um pouco mais, mas é mais completa e gera os gráficos
```