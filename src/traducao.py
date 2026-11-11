"""
Tradução do dataset agregado do OULAD para português (nomes de colunas e valores categóricos).

Uso direto:
    python traducao_oulad.py --entrada dataset_evasao.csv --saida dataset_evasao_pt.csv

Ou importando:
    from traducao_oulad import traduzir
    df_pt = traduzir(df)
"""
import argparse

import pandas as pd

# --------------------------------------------------------------------------
# Tipos de atividade do VLE (ambiente virtual de aprendizagem)
# --------------------------------------------------------------------------
TIPOS_ATIVIDADE = {
    "dataplus": "dataplus",
    "dualpane": "painel_duplo",
    "externalquiz": "quiz_externo",
    "forumng": "forum",
    "glossary": "glossario",
    "homepage": "pagina_inicial",
    "htmlactivity": "atividade_html",
    "oucollaborate": "ou_collaborate",
    "oucontent": "conteudo_ou",
    "ouelluminate": "ou_elluminate",
    "ouwiki": "ou_wiki",
    "page": "pagina",
    "questionnaire": "questionario",
    "quiz": "quiz",
    "repeatactivity": "atividade_repetida",
    "resource": "recurso",
    "sharedsubpage": "subpagina_compartilhada",
    "subpage": "subpagina",
    "url": "url",
}

# --------------------------------------------------------------------------
# Colunas
# --------------------------------------------------------------------------
COLUNAS = {
    # identificação / demografia
    "code_module": "codigo_modulo",
    "code_presentation": "codigo_oferta",
    "id_student": "id_estudante",
    "gender": "genero",
    "region": "regiao",
    "highest_education": "escolaridade_max",
    "imd_band": "faixa_imd",
    "age_band": "faixa_etaria",
    "num_of_prev_attempts": "num_tentativas_anteriores",
    "studied_credits": "creditos_cursados",
    "disability": "deficiencia",
    "final_result": "resultado_final",
    "date_registration": "data_registro",
    "module_presentation_length": "duracao_oferta_dias",
    # VLE
    "vle_total_clicks": "vle_total_cliques",
    "vle_dias_ativos": "vle_dias_ativos",
    "vle_sites_distintos": "vle_sites_distintos",
    "vle_primeiro_acesso": "vle_primeiro_acesso",
    "vle_ultimo_acesso": "vle_ultimo_acesso",
    "vle_media_clicks_por_dia_ativo": "vle_media_cliques_por_dia_ativo",
    "vle_clicks_pre_curso": "vle_cliques_pre_curso",
    "vle_clicks_ult_14d": "vle_cliques_ult_14d",
    "vle_clicks_ult_30d": "vle_cliques_ult_30d",
    "vle_dias_sem_acesso": "vle_dias_sem_acesso",
    "vle_tendencia_14d": "vle_tendencia_14d",
    **{f"vle_clicks_{k}": f"vle_cliques_{v}" for k, v in TIPOS_ATIVIDADE.items()},
    # avaliações (já em português; mantidas)
    # alvo e flags (já em português; mantidos)
}

# --------------------------------------------------------------------------
# Valores categóricos
# --------------------------------------------------------------------------
VALORES = {
    "genero": {"M": "Masculino", "F": "Feminino"},
    "deficiencia": {"N": "Não", "Y": "Sim"},
    "resultado_final": {
        "Pass": "Aprovado",
        "Distinction": "Aprovado com distinção",
        "Fail": "Reprovado",
        "Withdrawn": "Evadido",
    },
    "escolaridade_max": {
        "No Formal quals": "Sem qualificação formal",
        "Lower Than A Level": "Abaixo do A Level",
        "A Level or Equivalent": "A Level ou equivalente",
        "HE Qualification": "Ensino superior",
        "Post Graduate Qualification": "Pós-graduação",
    },
    "faixa_etaria": {"0-35": "Até 35", "35-55": "35 a 55", "55<=": "55 ou mais"},
    # o original traz "10-20" sem o símbolo %, padronizamos
    "faixa_imd": {"10-20": "10-20%"},
    "regiao": {
        "East Anglian Region": "Anglia Oriental",
        "East Midlands Region": "East Midlands",
        "Ireland": "Irlanda",
        "London Region": "Londres",
        "North Region": "Norte",
        "North Western Region": "Noroeste",
        "Scotland": "Escócia",
        "South East Region": "Sudeste",
        "South Region": "Sul",
        "South West Region": "Sudoeste",
        "Wales": "País de Gales",
        "West Midlands Region": "West Midlands",
        "Yorkshire Region": "Yorkshire",
    },
}

DESCRICOES = {
    "codigo_modulo": "Código (anonimizado) do módulo/disciplina",
    "codigo_oferta": "Código da oferta: ano + B (início em fevereiro) ou J (início em outubro)",
    "id_estudante": "Identificador do estudante",
    "genero": "Gênero do estudante",
    "regiao": "Região do Reino Unido onde o estudante vive",
    "escolaridade_max": "Maior nível de escolaridade na entrada do módulo",
    "faixa_imd": "Faixa do Índice de Privação Múltipla (IMD) da área de residência; 0-10% = mais carente",
    "faixa_etaria": "Faixa etária",
    "num_tentativas_anteriores": "Quantas vezes o estudante já tentou este módulo",
    "creditos_cursados": "Total de créditos dos módulos em que está matriculado",
    "deficiencia": "Declara alguma deficiência",
    "resultado_final": "Resultado final no módulo (NÃO usar como variável preditora)",
    "data_registro": "Dia da matrícula em relação ao início do curso (negativo = antes do início)",
    "duracao_oferta_dias": "Duração da oferta do módulo, em dias",
    "evasao": "VARIÁVEL-ALVO: 1 = evadiu (Withdrawn), 0 = permaneceu",
    "registrou_tarde": "1 se a matrícula ocorreu após o início do curso",
    "vle_total_cliques": "Total de cliques no VLE até o dia de corte",
    "vle_dias_ativos": "Número de dias distintos com acesso ao VLE",
    "vle_sites_distintos": "Número de recursos/páginas distintos acessados",
    "vle_primeiro_acesso": "Dia do primeiro acesso ao VLE",
    "vle_ultimo_acesso": "Dia do último acesso ao VLE até o corte",
    "vle_media_cliques_por_dia_ativo": "Média de cliques por dia ativo",
    "vle_cliques_pre_curso": "Cliques antes do início do curso (dia < 0)",
    "vle_cliques_ult_14d": "Cliques nos 14 dias anteriores ao corte",
    "vle_cliques_ult_30d": "Cliques nos 30 dias anteriores ao corte",
    "vle_dias_sem_acesso": "Dias desde o último acesso até o corte",
    "vle_tendencia_14d": "Proporção dos cliques totais concentrada nos últimos 14 dias",
    "aval_n_entregues": "Avaliações entregues até o corte",
    "aval_score_medio": "Nota média nas avaliações entregues",
    "aval_score_min": "Menor nota nas avaliações entregues",
    "aval_score_ponderado_soma": "Soma das notas ponderadas pelo peso de cada avaliação",
    "aval_n_atrasadas": "Avaliações entregues após o prazo",
    "aval_atraso_medio": "Atraso médio (dias) em relação ao prazo; negativo = adiantado",
    "aval_n_banked": "Avaliações aproveitadas de uma tentativa anterior (banked)",
    "aval_n_devidas": "Avaliações com prazo até o corte",
    "aval_taxa_entrega": "Entregues ÷ devidas (limitado a 1)",
    "nunca_acessou_vle": "1 se não houve nenhum clique no VLE até o corte",
}
for _k, _v in TIPOS_ATIVIDADE.items():
    DESCRICOES[f"vle_cliques_{_v}"] = f"Cliques em atividades do tipo '{_k}'"


def traduzir(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma cópia do dataframe com colunas e categorias em português."""
    out = df.rename(columns=COLUNAS).copy()
    for col, mapa in VALORES.items():
        if col in out.columns:
            out[col] = out[col].replace(mapa)
    return out


def dicionario_de_dados(df_pt: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "coluna": df_pt.columns,
            "tipo": [str(t) for t in df_pt.dtypes],
            "descricao": [DESCRICOES.get(c, "") for c in df_pt.columns],
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="dataset_evasao.csv")
    ap.add_argument("--saida", default="dataset_evasao_pt.csv")
    ap.add_argument("--dicionario", default="dicionario_dados.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.entrada)
    pt = traduzir(df)
    pt.to_csv(a.saida, index=False)
    dicionario_de_dados(pt).to_csv(a.dicionario, index=False)
    print(f"Salvo: {a.saida} {pt.shape} | {a.dicionario}")
