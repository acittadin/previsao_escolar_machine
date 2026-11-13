import pandas as pd

from preprocessamento import ALVO, COL_ID

# Análise exploratória em forma de tabelas impressas (os gráficos do relatório foram gerados no projeto
# antigo, em src/eda.py). Trabalha com o dataset BRUTO (antes do tratamento), para ler as categorias originais.

CATEGORICAS = ["codigo_modulo", "codigo_oferta", "escolaridade_max", "faixa_imd", "faixa_etaria", "genero",
               "deficiencia", "num_tentativas_anteriores", "regiao"]
NUMERICAS_CHAVE = ["vle_total_cliques", "vle_dias_ativos", "vle_cliques_ult_14d", "vle_dias_sem_acesso",
                   "aval_n_entregues", "aval_score_medio", "aval_taxa_entrega", "creditos_cursados"]


def visaoGeral(df):
    print(f"Linhas: {len(df)} | Colunas: {df.shape[1]} | Estudantes distintos: {df[COL_ID].nunique()}")
    print(f"Taxa de evasao: {df[ALVO].mean():.2%}  (classe desbalanceada: sempre prever 'permanece' acerta {1 - df[ALVO].mean():.1%})")
    print("\nResultado final:")
    print(df["resultado_final"].value_counts().to_string())

    nulos = (df.isna().mean() * 100).round(1)
    print("\nColunas com valores ausentes (%):")
    print(nulos[nulos > 0].sort_values(ascending=False).to_string())
    print("\nAssimetria (skew) dos cliques: ", round(df["vle_total_cliques"].skew(), 2),
          "(valores > 1 indicam cauda longa à direita: motivo do log)")


def taxaPorCategoria(df, colunas=CATEGORICAS):
    for coluna in colunas:
        tabela = df.groupby(coluna)[ALVO].agg(["count", "mean"]).sort_values("mean", ascending=False)
        tabela.columns = ["n", "evasao_%"]
        tabela["evasao_%"] = (tabela["evasao_%"] * 100).round(1)
        print(f"\n--- Evasao por {coluna} ---")
        print(tabela.to_string())


def compararClasses(df, colunas=NUMERICAS_CHAVE):
    tabela = df.groupby(ALVO)[colunas].agg(["mean", "median"]).T.round(2)
    tabela.columns = ["permanece", "evade"]
    print("\n--- Media e mediana por classe ---")
    print(tabela.to_string())


def correlacaoComAlvo(df, quantidade=8):
    # Spearman: mede associação monotônica, não é afetada pela assimetria dos cliques
    numericas = df.select_dtypes("number").drop(columns=[COL_ID, ALVO])
    corr = numericas.apply(lambda s: s.corr(df[ALVO], method="spearman")).dropna().sort_values()
    print(f"\n--- Correlacao de Spearman com a evasao ({quantidade} maiores em cada sentido) ---")
    print("Reduzem o risco:\n" + corr.head(quantidade).round(3).to_string())
    print("\nAumentam o risco:\n" + corr.tail(quantidade).round(3).to_string())


def analisarDados(caminho):
    df = pd.read_csv(caminho)
    visaoGeral(df)
    taxaPorCategoria(df)
    compararClasses(df)
    correlacaoComAlvo(df)
