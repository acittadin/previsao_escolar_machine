import pandas as pd
from sklearn.model_selection import train_test_split


def carregarDados(caminho):
    df = pd.read_csv(caminho)

    # ----------------------------------------------------------------------------------------------------
    # LIMPEZA DE DADOS
    # ----------------------------------------------------------------------------------------------------

    # print(df.head())
    # print(df.info())
    print(df["evasao"].value_counts(normalize=True))

    df = df.drop_duplicates(subset=["id_estudante", "codigo_modulo", "codigo_oferta"])

    # resultado_final é o próprio desfecho (vazamento) e vle_ultimo_acesso é redundante
    # id_estudante e codigo_oferta só identificam a linha: não entram como característica
    df = df.drop(columns=["resultado_final", "vle_ultimo_acesso", "id_estudante", "codigo_oferta"])

    # -----------------------------------------------------------------------------------------------------
    # TRATAMENTO DE DADOS
    # -----------------------------------------------------------------------------------------------------

    # nulos viram 0 (nota ausente = 0, sem acesso = 0...). Simples, mas mistura "sem dado" com "zero de verdade"
    df = df.fillna(0)

    # categorias de texto viram colunas 0/1 (One-Hot)
    colunas_texto = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=colunas_texto, drop_first=True, dtype=int)

    # -----------------------------------------------------------------------------------------------------
    # Divisão de X e Y
    # -----------------------------------------------------------------------------------------------------

    X = df.drop(columns=["evasao"])
    y = df["evasao"]

    # Sorteio ALEATÓRIO 70/30, como no projeto de diabetes. Aqui mora o problema desta versão:
    # estudantes repetidos e ofertas diferentes se misturam entre treino e teste.
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
