import numpy as np
import pandas as pd
from src import config as C

def limpar_dados(df, cutoff=C.CUTOFF_DIAS):
    """
    Limpa o dataset aplicando regras determinísticas (sem vazamento de dados).
    Gera flags de ausência, transforma variáveis categóricas e aplica log1p nos cliques.
    """
    print(f"Iniciando limpeza... Linhas originais: {len(df)}")
    df = df.copy()

    # 1. Remover duplicatas exatas pela chave primária
    chave = ["id_estudante", "codigo_modulo", "codigo_oferta"]
    df = df.drop_duplicates(subset=chave)
    print(f"Linhas após remoção de duplicatas: {len(df)}")

    # 2. Remover colunas que vazam a resposta ou são 100% redundantes
    vazamentos = ["resultado_final", "vle_ultimo_acesso"]
    df = df.drop(columns=[c for c in vazamentos if c in df.columns], errors='ignore')

    # 3. Criar flags para ausências importantes (a ausência de nota/IMD é um sinal forte)
    df["faixa_imd_ausente"] = df["faixa_imd"].isna().astype(int)
    df["aval_sem_nota"] = df["aval_score_medio"].isna().astype(int)

    # 4. Preencher nulos com significado de negócio (alunos que nunca acessaram o VLE)
    df["vle_media_cliques_por_dia_ativo"] = df["vle_media_cliques_por_dia_ativo"].fillna(0)
    df["vle_tendencia_14d"] = df["vle_tendencia_14d"].fillna(0)
    df["vle_dias_sem_acesso"] = df["vle_dias_sem_acesso"].fillna(cutoff)
    df["vle_primeiro_acesso"] = df["vle_primeiro_acesso"].fillna(cutoff)

    # 5. Converter categorias textuais em números (Encoding Ordinal e Binário)
    mapa_escolaridade = {k: i for i, k in enumerate([
        "Sem qualificação formal", "Abaixo do A Level", "A Level ou equivalente", 
        "Ensino superior", "Pós-graduação"
    ])}
    mapa_imd = {k: i for i, k in enumerate([
        "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", 
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
    ])}
    mapa_idade = {k: i for i, k in enumerate(["Até 35", "35 a 55", "55 ou mais"])}

    df["escolaridade_ord"] = df["escolaridade_max"].map(mapa_escolaridade)
    df["imd_ord"] = df["faixa_imd"].map(mapa_imd)
    df["faixa_etaria_ord"] = df["faixa_etaria"].map(mapa_idade)
    df["genero_masc"] = (df["genero"] == "Masculino").astype(int)
    df["deficiencia_sim"] = (df["deficiencia"] == "Sim").astype(int)

    df = df.drop(columns=["escolaridade_max", "faixa_imd", "faixa_etaria", "genero", "deficiencia"])

    # 6. Aplicar log1p nos cliques para reduzir a assimetria (~3) e ajudar o K-Means e SVM
    colunas_cliques = [c for c in df.columns if c.startswith("vle_cliques_") or c in ("vle_total_cliques", "vle_media_cliques_por_dia_ativo")]
    
    for col in colunas_cliques:
        df[f"log_{col}"] = np.log1p(df[col])
        
    df = df.drop(columns=colunas_cliques)

    # 7. Ajustes finais de tipo
    df["registrou_tarde"] = df["registrou_tarde"].astype(float)
    
    print(f"Limpeza concluída. Formato final: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df


def reverter_log_cliques(df):
    """Reverte o log1p para a análise de perfis do K-Means ficar na escala original de cliques."""
    df_revertido = df.copy()
    colunas_log = [c for c in df_revertido.columns if c.startswith("log_")]
    
    for col in colunas_log:
        nome_original = col.replace("log_", "", 1)
        df_revertido[nome_original] = np.expm1(df_revertido[col])
        
    return df_revertido


def colunas_modelo(df):
    """Separa rapidamente as colunas numéricas das categóricas para os pipelines."""
    ignorar = {C.ALVO, C.COL_ID, C.COL_OFERTA, *C.CAT_COLS}
    num = [c for c in df.columns if c not in ignorar and pd.api.types.is_numeric_dtype(df[c])]
    cat = list(C.CAT_COLS)
    return num, cat


def split_temporal(df):
    """
    Separa treino e teste usando o tempo (ofertas). 
    Treina com o passado e testa com a oferta mais recente para simular o uso real.
    """
    teste = df[df[C.COL_OFERTA] == C.OFERTA_TESTE].reset_index(drop=True)
    treino = df[df[C.COL_OFERTA] != C.OFERTA_TESTE].reset_index(drop=True)
    return treino, teste


def dividir_treino_teste(df):
    """Alias compatível com o restante do projeto."""
    return split_temporal(df)


def reverter_log(df):
    """Alias compatível com o restante do projeto."""
    return reverter_log_cliques(df)


def main():
    df_bruto = pd.read_csv(C.DATASET_PT)
    df_limpo = limpar_dados(df_bruto)
    df_limpo.to_csv(C.DATASET_LIMPO, index=False)

    treino, teste = dividir_treino_teste(df_limpo)
    print(f"\nDistribuição final:")
    print(f"Treino: {len(treino)} alunos ({treino[C.ALVO].mean():.1%} de evasão)")
    print(f"Teste ({C.OFERTA_TESTE}): {len(teste)} alunos ({teste[C.ALVO].mean():.1%} de evasão)")


if __name__ == "__main__":
    main()