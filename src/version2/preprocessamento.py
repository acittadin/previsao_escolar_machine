import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALVO = "evasao"
COL_ID = "id_estudante"
COL_OFERTA = "codigo_oferta"
CHAVE = ["id_estudante", "codigo_modulo", "codigo_oferta"]
COLUNAS_CATEGORICAS = ["regiao", "codigo_modulo"]    # viram One-Hot; todas as outras são numéricas

CORTE_DIAS = 90               # momento do curso em que a previsão é feita (só dados até esse dia)
OFERTA_TESTE = "2014J"        # teste = oferta mais recente (split TEMPORAL, não aleatório)

# categorias com ordem natural (da menor para a maior)
ORDEM_ESCOLARIDADE = ["Sem qualificação formal", "Abaixo do A Level", "A Level ou equivalente",
                      "Ensino superior", "Pós-graduação"]
ORDEM_IMD = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
ORDEM_IDADE = ["Até 35", "35 a 55", "55 ou mais"]


def colunasNumericas(X):
    # Usada pelos pipelines de todos os modelos (o ColumnTransformer aceita uma função no lugar da lista).
    # Funciona mesmo depois de remover colunas na seleção de atributos.
    return [coluna for coluna in X.columns if coluna not in COLUNAS_CATEGORICAS]


def colunasCategoricas(X):
    return [coluna for coluna in COLUNAS_CATEGORICAS if coluna in X.columns]


def criarPreparo(escalar):
    # Imputação (mediana) e padronização são APRENDIDAS dentro do pipeline: em cada rodada da validação
    # cruzada elas usam só o treino daquela rodada. Se fossem feitas antes, o teste vazaria para o treino.
    # Modelos de árvore não precisam de escala (escalar=False); SVM e regressão logística precisam.
    numericas = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler() if escalar else "passthrough"),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numericas, colunasNumericas),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), colunasCategoricas),
        ],
        verbose_feature_names_out=False  # mantém os nomes originais
    )


def reverterLog(X):
    # Devolve os cliques na escala original (as colunas log_* viram vle_* de novo). Usado nos perfis.
    out = X.copy()
    for coluna in [c for c in out.columns if c.startswith("log_")]:
        out[coluna.replace("log_", "", 1)] = np.expm1(out[coluna])
    return out


def carregarDados(caminho, corte=CORTE_DIAS):
    df = pd.read_csv(caminho)

    # ----------------------------------------------------------------------------------------------------
    # LIMPEZA DE DADOS
    # ----------------------------------------------------------------------------------------------------

    # print(df.head())
    # print(df.info())
    # print(df.describe())
    print(df[ALVO].value_counts(normalize=True))
    # só ~12% dos estudantes evadem: o dataset é DESBALANCEADO.
    # Um modelo que sempre responde "não evade" acerta ~88%, então a acurácia sozinha engana.
    # Por isso a avaliação usa PR-AUC, ROC-AUC, recall e lift (e não só acurácia).

    assert set(df[ALVO].unique()) <= {0, 1}, "o alvo deve ser binário (0/1)"

    duplicados = df.duplicated(subset=CHAVE).sum()
    print("Duplicados (estudante + módulo + oferta):", duplicados)
    df = df.drop_duplicates(subset=CHAVE).reset_index(drop=True)

    # remover colunas que VAZAM o alvo ou são redundantes
    #   resultado_final : é o próprio desfecho (Evadido = alvo)
    #   vle_ultimo_acesso : = corte - vle_dias_sem_acesso (correlação -1)
    for coluna in ["resultado_final", "vle_ultimo_acesso"]:
        if coluna in df.columns:
            df = df.drop(columns=[coluna])

    # -----------------------------------------------------------------------------------------------------
    # TRATAMENTO DE DADOS
    # -----------------------------------------------------------------------------------------------------

    # a ausência também é informação (ex.: quem não tem nota é quem não entregou nada): vira coluna 0/1
    df["faixa_imd_ausente"] = df["faixa_imd"].isna().astype(int)
    df["aval_sem_nota"] = df["aval_score_medio"].isna().astype(int)

    # ausências com significado conhecido: quem nunca acessou o VLE não tem média/tendência de cliques
    # e ficou "sem acesso" durante toda a janela
    df["vle_media_cliques_por_dia_ativo"] = df["vle_media_cliques_por_dia_ativo"].fillna(0)
    df["vle_tendencia_14d"] = df["vle_tendencia_14d"].fillna(0)
    df["vle_dias_sem_acesso"] = df["vle_dias_sem_acesso"].fillna(corte)
    df["vle_primeiro_acesso"] = df["vle_primeiro_acesso"].fillna(corte)
    # os demais nulos (notas, atraso médio, faixa_imd...) são imputados pela mediana DENTRO do pipeline

    # variáveis com ordem natural viram número (0, 1, 2...); gênero e deficiência viram 0/1
    df["escolaridade_ord"] = df["escolaridade_max"].map({k: i for i, k in enumerate(ORDEM_ESCOLARIDADE)})
    df["imd_ord"] = df["faixa_imd"].map({k: i for i, k in enumerate(ORDEM_IMD)})
    df["faixa_etaria_ord"] = df["faixa_etaria"].map({k: i for i, k in enumerate(ORDEM_IDADE)})
    df["genero_masc"] = (df["genero"] == "Masculino").astype(int)
    df["deficiencia_sim"] = (df["deficiencia"] == "Sim").astype(int)
    df = df.drop(columns=["escolaridade_max", "faixa_imd", "faixa_etaria", "genero", "deficiencia"])
    assert df["escolaridade_ord"].notna().all() and df["faixa_etaria_ord"].notna().all(), "categoria inesperada"

    # cliques são muito assimétricos (assimetria ~3): log(1 + x) ajuda SVM e regressão (não muda as árvores)
    cliques = [c for c in df.columns if c.startswith("vle_cliques_")
               or c in ("vle_total_cliques", "vle_media_cliques_por_dia_ativo")]
    for coluna in cliques:
        df[f"log_{coluna}"] = np.log1p(df[coluna])
    df = df.drop(columns=cliques)

    df["registrou_tarde"] = df["registrou_tarde"].astype(float)

    # -----------------------------------------------------------------------------------------------------
    # Divisão de X e Y
    # -----------------------------------------------------------------------------------------------------

    # Split TEMPORAL: treina com as ofertas antigas e testa na mais recente (como seria no uso real:
    # treinar com o passado e prever a oferta seguinte). Um sorteio aleatório misturaria ofertas e
    # estudantes repetidos entre treino e teste e deixaria o resultado otimista demais.
    treino = df[df[COL_OFERTA] != OFERTA_TESTE].reset_index(drop=True)
    teste = df[df[COL_OFERTA] == OFERTA_TESTE].reset_index(drop=True)

    # o mesmo estudante pode aparecer em ofertas diferentes: na validação cruzada os "grupos" garantem
    # que todas as linhas de um estudante caiam no mesmo fold
    grupos = treino[COL_ID].values

    descartar = [ALVO, COL_ID, COL_OFERTA]
    X_train, y_train = treino.drop(columns=descartar), treino[ALVO]
    X_test, y_test = teste.drop(columns=descartar), teste[ALVO]

    return X_train, X_test, y_train, y_test, grupos
