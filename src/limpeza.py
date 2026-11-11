"""
Limpeza e preparação dos dados.

Regras DETERMINÍSTICAS (valem para qualquer split) ficam aqui e geram data/processed/dataset_limpo.csv.
Regras que APRENDEM com os dados (imputação por mediana, escala, one-hot) ficam dentro dos
pipelines do scikit-learn e são ajustadas somente no treino, para não vazar informação.

Uso:  python -m src.limpeza
"""
import numpy as np
import pandas as pd

from src import config as C

# Escala ordinal (menor -> maior)
ORDEM_ESCOLARIDADE = [
    "Sem qualificação formal", "Abaixo do A Level", "A Level ou equivalente",
    "Ensino superior", "Pós-graduação",
]
ORDEM_IMD = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%",
             "60-70%", "70-80%", "80-90%", "90-100%"]
ORDEM_IDADE = ["Até 35", "35 a 55", "55 ou mais"]

# Colunas redundantes (correlação ~1 com outra) ou que vazam o alvo
COLS_VAZAMENTO = ["resultado_final"]
COLS_REDUNDANTES = ["vle_ultimo_acesso"]        # = cutoff - vle_dias_sem_acesso (rho = -1)

CHAVE = ["id_estudante", "codigo_modulo", "codigo_oferta"]


def colunas_cliques(df: pd.DataFrame) -> list[str]:
    """Colunas de contagem de cliques (muito assimétricas -> log1p)."""
    return [c for c in df.columns if c.startswith("vle_cliques_") or c in
            ("vle_total_cliques", "vle_media_cliques_por_dia_ativo")]


def limpar(df: pd.DataFrame, cutoff: int = C.CUTOFF_DIAS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica a limpeza e devolve (df_limpo, relatorio). `cutoff` = dia de corte usado na agregação."""
    log = []
    n0 = len(df)
    df = df.copy()

    # 1) duplicatas na chave (estudante x módulo x oferta)
    dup = df.duplicated(subset=CHAVE).sum()
    df = df.drop_duplicates(subset=CHAVE)
    log.append(("Duplicatas removidas (chave estudante+módulo+oferta)", int(dup)))

    # 2) alvo válido
    assert set(df[C.ALVO].unique()) <= {0, 1}, "alvo deve ser binário"

    # 3) vazamento / redundância
    df = df.drop(columns=[c for c in COLS_VAZAMENTO + COLS_REDUNDANTES if c in df.columns])
    log.append(("Colunas removidas (vazamento/redundância)", ", ".join(COLS_VAZAMENTO + COLS_REDUNDANTES)))

    # 4) flags de ausência (a ausência em si é informativa)
    df["faixa_imd_ausente"] = df["faixa_imd"].isna().astype(int)
    df["aval_sem_nota"] = df["aval_score_medio"].isna().astype(int)
    log.append(("Flag faixa_imd_ausente = 1", int(df["faixa_imd_ausente"].sum())))
    log.append(("Flag aval_sem_nota = 1", int(df["aval_sem_nota"].sum())))

    # 5) ausências com significado conhecido -> valor determinístico
    #    (quem nunca acessou o VLE: 0 cliques/média e "sem acesso" durante toda a janela)
    df["vle_media_cliques_por_dia_ativo"] = df["vle_media_cliques_por_dia_ativo"].fillna(0)
    df["vle_tendencia_14d"] = df["vle_tendencia_14d"].fillna(0)
    df["vle_dias_sem_acesso"] = df["vle_dias_sem_acesso"].fillna(cutoff)
    df["vle_primeiro_acesso"] = df["vle_primeiro_acesso"].fillna(cutoff)

    # 6) codificação de variáveis categóricas
    df["escolaridade_ord"] = df["escolaridade_max"].map({k: i for i, k in enumerate(ORDEM_ESCOLARIDADE)})
    df["imd_ord"] = df["faixa_imd"].map({k: i for i, k in enumerate(ORDEM_IMD)})   # NaN -> imputado depois
    df["faixa_etaria_ord"] = df["faixa_etaria"].map({k: i for i, k in enumerate(ORDEM_IDADE)})
    df["genero_masc"] = (df["genero"] == "Masculino").astype(int)
    df["deficiencia_sim"] = (df["deficiencia"] == "Sim").astype(int)
    df = df.drop(columns=["escolaridade_max", "faixa_imd", "faixa_etaria", "genero", "deficiencia"])
    for c in ("escolaridade_ord", "faixa_etaria_ord"):
        assert df[c].notna().all(), f"categoria inesperada em {c}"

    # 7) transformação log1p nas colunas de cliques (assimetria ~3). Não altera árvores,
    #    ajuda SVM e KMeans. Os nomes são mantidos; o prefixo 'log_' sinaliza a transformação.
    cliques = colunas_cliques(df)
    for c in cliques:
        df[f"log_{c}"] = np.log1p(df[c])
    df = df.drop(columns=cliques)
    log.append(("Colunas com log1p", len(cliques)))

    # 8) tipos
    df["registrou_tarde"] = df["registrou_tarde"].astype(float)

    log.append(("Linhas antes", n0))
    log.append(("Linhas depois", len(df)))
    log.append(("Colunas depois", df.shape[1]))
    nulos = df.isna().sum()
    log.append(("Nulos restantes (imputados no pipeline)",
                "; ".join(f"{k}={v}" for k, v in nulos[nulos > 0].items()) or "nenhum"))
    return df, pd.DataFrame(log, columns=["etapa", "resultado"])


def reverter_log(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve os cliques na escala original (para interpretar perfis)."""
    out = df.copy()
    for c in [c for c in out.columns if c.startswith("log_")]:
        out[c.replace("log_", "", 1)] = np.expm1(out[c])
    return out


def colunas_modelo(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numéricas, categóricas) usadas pelos modelos."""
    excluir = {C.ALVO, C.COL_ID, C.COL_OFERTA, *C.CAT_COLS}
    num = [c for c in df.columns if c not in excluir and pd.api.types.is_numeric_dtype(df[c])]
    return num, list(C.CAT_COLS)


def dividir_treino_teste(df: pd.DataFrame):
    """
    Split TEMPORAL: treino = ofertas anteriores a OFERTA_TESTE; teste = OFERTA_TESTE.
    Simula o uso real (treinar com o passado, prever a oferta seguinte).
    Um mesmo estudante pode aparecer em ofertas diferentes (histórico passado é legítimo),
    mas na validação cruzada do treino os folds são agrupados por estudante (GroupKFold).
    Devolve (df_treino, df_teste).
    """
    teste = df[df[C.COL_OFERTA] == C.OFERTA_TESTE]
    treino = df[df[C.COL_OFERTA] != C.OFERTA_TESTE]
    return treino.reset_index(drop=True), teste.reset_index(drop=True)


def main():
    df = pd.read_csv(C.DATASET_PT)
    limpo, rel = limpar(df)
    limpo.to_csv(C.DATASET_LIMPO, index=False)
    rel.to_csv(C.TABELAS / "relatorio_limpeza.csv", index=False)
    print(rel.to_string(index=False))
    tr, te = dividir_treino_teste(limpo)
    print(f"\nTreino: {len(tr)} ({tr[C.ALVO].mean():.1%} evasão) | "
          f"Teste ({C.OFERTA_TESTE}): {len(te)} ({te[C.ALVO].mean():.1%} evasão)")
    print(f"Salvo: {C.DATASET_LIMPO}")


if __name__ == "__main__":
    main()
