# é NECESSÁRIO rodar esse script apenas uma vez para gerar o dataset agregado (data/oulad_evasao.csv)
# a partir dos 7 CSVs originais do OULAD (Open University Learning Analytics Dataset).
#
# 1. baixe o dataset (archive.zip / https://analyse.kmi.open.ac.uk/open_dataset) e extraia os CSVs em data/raw/
# 2. rode, a partir da RAIZ do projeto:
#       python setup/agregarOulad.py                              # corte de 90 dias -> data/oulad_evasao.csv
#       python setup/agregarOulad.py --cortes 0 30 60 90 120 180  # também gera data/cortes/oulad_corte_N.csv
#
# Unidade de análise: 1 linha = 1 estudante em 1 oferta de módulo (id_estudante, codigo_modulo, codigo_oferta)
# Alvo: evasao = 1 se o resultado final foi "Withdrawn" (cancelou a matrícula), senão 0
#
# O CORTE (em dias) é o momento do curso em que a previsão é feita: só entram eventos até esse dia.
# Quem já cancelou a matrícula até o corte é removido (a evasão deles já é conhecida, não há o que prever).

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHAVE = ["code_module", "code_presentation", "id_student"]

# ------------------------------------------------------------------------------------------------
# TRADUÇÃO (nomes de colunas e categorias para português)
# ------------------------------------------------------------------------------------------------

TIPOS_ATIVIDADE = {'dataplus': 'dataplus',
 'dualpane': 'painel_duplo',
 'externalquiz': 'quiz_externo',
 'forumng': 'forum',
 'glossary': 'glossario',
 'homepage': 'pagina_inicial',
 'htmlactivity': 'atividade_html',
 'oucollaborate': 'ou_collaborate',
 'oucontent': 'conteudo_ou',
 'ouelluminate': 'ou_elluminate',
 'ouwiki': 'ou_wiki',
 'page': 'pagina',
 'questionnaire': 'questionario',
 'quiz': 'quiz',
 'repeatactivity': 'atividade_repetida',
 'resource': 'recurso',
 'sharedsubpage': 'subpagina_compartilhada',
 'subpage': 'subpagina',
 'url': 'url'}

COLUNAS = {'code_module': 'codigo_modulo',
 'code_presentation': 'codigo_oferta',
 'id_student': 'id_estudante',
 'gender': 'genero',
 'region': 'regiao',
 'highest_education': 'escolaridade_max',
 'imd_band': 'faixa_imd',
 'age_band': 'faixa_etaria',
 'num_of_prev_attempts': 'num_tentativas_anteriores',
 'studied_credits': 'creditos_cursados',
 'disability': 'deficiencia',
 'final_result': 'resultado_final',
 'date_registration': 'data_registro',
 'module_presentation_length': 'duracao_oferta_dias',
 'vle_total_clicks': 'vle_total_cliques',
 'vle_dias_ativos': 'vle_dias_ativos',
 'vle_sites_distintos': 'vle_sites_distintos',
 'vle_primeiro_acesso': 'vle_primeiro_acesso',
 'vle_ultimo_acesso': 'vle_ultimo_acesso',
 'vle_media_clicks_por_dia_ativo': 'vle_media_cliques_por_dia_ativo',
 'vle_clicks_pre_curso': 'vle_cliques_pre_curso',
 'vle_clicks_ult_14d': 'vle_cliques_ult_14d',
 'vle_clicks_ult_30d': 'vle_cliques_ult_30d',
 'vle_dias_sem_acesso': 'vle_dias_sem_acesso',
 'vle_tendencia_14d': 'vle_tendencia_14d',
 'vle_clicks_dataplus': 'vle_cliques_dataplus',
 'vle_clicks_dualpane': 'vle_cliques_painel_duplo',
 'vle_clicks_externalquiz': 'vle_cliques_quiz_externo',
 'vle_clicks_forumng': 'vle_cliques_forum',
 'vle_clicks_glossary': 'vle_cliques_glossario',
 'vle_clicks_homepage': 'vle_cliques_pagina_inicial',
 'vle_clicks_htmlactivity': 'vle_cliques_atividade_html',
 'vle_clicks_oucollaborate': 'vle_cliques_ou_collaborate',
 'vle_clicks_oucontent': 'vle_cliques_conteudo_ou',
 'vle_clicks_ouelluminate': 'vle_cliques_ou_elluminate',
 'vle_clicks_ouwiki': 'vle_cliques_ou_wiki',
 'vle_clicks_page': 'vle_cliques_pagina',
 'vle_clicks_questionnaire': 'vle_cliques_questionario',
 'vle_clicks_quiz': 'vle_cliques_quiz',
 'vle_clicks_repeatactivity': 'vle_cliques_atividade_repetida',
 'vle_clicks_resource': 'vle_cliques_recurso',
 'vle_clicks_sharedsubpage': 'vle_cliques_subpagina_compartilhada',
 'vle_clicks_subpage': 'vle_cliques_subpagina',
 'vle_clicks_url': 'vle_cliques_url'}

VALORES = {'genero': {'M': 'Masculino', 'F': 'Feminino'},
 'deficiencia': {'N': 'Não', 'Y': 'Sim'},
 'resultado_final': {'Pass': 'Aprovado',
                     'Distinction': 'Aprovado com distinção',
                     'Fail': 'Reprovado',
                     'Withdrawn': 'Evadido'},
 'escolaridade_max': {'No Formal quals': 'Sem qualificação formal',
                      'Lower Than A Level': 'Abaixo do A Level',
                      'A Level or Equivalent': 'A Level ou equivalente',
                      'HE Qualification': 'Ensino superior',
                      'Post Graduate Qualification': 'Pós-graduação'},
 'faixa_etaria': {'0-35': 'Até 35', '35-55': '35 a 55', '55<=': '55 ou mais'},
 'faixa_imd': {'10-20': '10-20%'},
 'regiao': {'East Anglian Region': 'Anglia Oriental',
            'East Midlands Region': 'East Midlands',
            'Ireland': 'Irlanda',
            'London Region': 'Londres',
            'North Region': 'Norte',
            'North Western Region': 'Noroeste',
            'Scotland': 'Escócia',
            'South East Region': 'Sudeste',
            'South Region': 'Sul',
            'South West Region': 'Sudoeste',
            'Wales': 'País de Gales',
            'West Midlands Region': 'West Midlands',
            'Yorkshire Region': 'Yorkshire'}}


def traduzir(df):
    df = df.rename(columns=COLUNAS).copy()
    for coluna, mapa in VALORES.items():
        if coluna in df.columns:
            df[coluna] = df[coluna].replace(mapa)
    return df


# ------------------------------------------------------------------------------------------------
# AGREGAÇÃO
# ------------------------------------------------------------------------------------------------

def carregarBrutos(pasta):
    pasta = Path(pasta)
    d = {
        "courses": pd.read_csv(pasta / "courses.csv"),
        "assessments": pd.read_csv(pasta / "assessments.csv"),
        "student_assessment": pd.read_csv(pasta / "studentAssessment.csv"),
        "student_info": pd.read_csv(pasta / "studentInfo.csv"),
        "student_reg": pd.read_csv(pasta / "studentRegistration.csv"),
        "vle": pd.read_csv(pasta / "vle.csv"),
    }
    # a tabela studentVle tem ~10,6 milhões de linhas: tipos menores para economizar memória
    d["student_vle"] = pd.read_csv(
        pasta / "studentVle.csv",
        dtype={"code_module": "category", "code_presentation": "category", "id_student": "int32",
               "id_site": "int32", "date": "int16", "sum_click": "int16"},
    )
    return d


def baseEstudantes(d, corte):
    base = d["student_info"].merge(d["student_reg"], on=CHAVE, how="left")
    base = base.merge(d["courses"], on=["code_module", "code_presentation"], how="left")
    base["evasao"] = (base["final_result"] == "Withdrawn").astype(int)

    ja_saiu = base["date_unregistration"].notna() & (base["date_unregistration"] <= corte)
    print(f"  removendo {ja_saiu.sum()} estudantes que já cancelaram até o dia {corte}")
    base = base[~ja_saiu].copy()

    # date_registration é conhecida antes do início do curso (negativo = dias antes do início)
    base["registrou_tarde"] = (base["date_registration"] > 0).astype(float)
    base.loc[base["date_registration"].isna(), "registrou_tarde"] = np.nan
    return base


def featuresVLE(d, corte):
    cliques = d["student_vle"]
    cliques = cliques[cliques["date"] <= corte]

    # junta o tipo de atividade (forumng, quiz, resource, ...)
    vle = d["vle"][["id_site", "code_module", "code_presentation", "activity_type"]]
    cliques = cliques.merge(vle, on=["id_site", "code_module", "code_presentation"], how="left")
    cliques["code_module"] = cliques["code_module"].astype(str)
    cliques["code_presentation"] = cliques["code_presentation"].astype(str)

    g = cliques.groupby(CHAVE, observed=True)
    out = g.agg(
        vle_total_clicks=("sum_click", "sum"),
        vle_dias_ativos=("date", "nunique"),
        vle_sites_distintos=("id_site", "nunique"),
        vle_primeiro_acesso=("date", "min"),
        vle_ultimo_acesso=("date", "max"),
    )
    out["vle_media_clicks_por_dia_ativo"] = out["vle_total_clicks"] / out["vle_dias_ativos"]

    # cliques antes do início do curso (date < 0) e nas janelas mais recentes
    pre = cliques[cliques["date"] < 0].groupby(CHAVE, observed=True)["sum_click"].sum().rename("vle_clicks_pre_curso")
    out = out.join(pre)
    for janela in (14, 30):
        recentes = cliques[cliques["date"] > corte - janela].groupby(CHAVE, observed=True)["sum_click"].sum()
        out = out.join(recentes.rename(f"vle_clicks_ult_{janela}d"))
        out[f"vle_clicks_ult_{janela}d"] = out[f"vle_clicks_ult_{janela}d"].fillna(0)
    out["vle_dias_sem_acesso"] = corte - out["vle_ultimo_acesso"]
    out["vle_tendencia_14d"] = out["vle_clicks_ult_14d"] / (out["vle_total_clicks"] + 1)

    # cliques por tipo de atividade
    por_tipo = cliques.groupby(CHAVE + ["activity_type"], observed=True)["sum_click"].sum().unstack(fill_value=0)
    por_tipo.columns = [f"vle_clicks_{c}" for c in por_tipo.columns]
    out = out.join(por_tipo).reset_index()

    cols_zero = [c for c in out.columns if c.startswith("vle_clicks") or c == "vle_total_clicks"]
    out[cols_zero] = out[cols_zero].fillna(0)
    return out


def featuresAvaliacoes(d, corte):
    a = d["assessments"]
    sa = d["student_assessment"].merge(a, on="id_assessment", how="left")
    sa = sa[sa["date_submitted"] <= corte]

    sa["atraso_dias"] = sa["date_submitted"] - sa["date"]          # date = prazo (NaN em alguns exames finais)
    sa["entregue_atrasada"] = (sa["atraso_dias"] > 0).astype(int)
    sa["score_ponderado"] = sa["score"] * sa["weight"] / 100.0

    out = sa.groupby(CHAVE).agg(
        aval_n_entregues=("id_assessment", "nunique"),
        aval_score_medio=("score", "mean"),
        aval_score_min=("score", "min"),
        aval_score_ponderado_soma=("score_ponderado", "sum"),
        aval_n_atrasadas=("entregue_atrasada", "sum"),
        aval_atraso_medio=("atraso_dias", "mean"),
        aval_n_banked=("is_banked", "sum"),
    ).reset_index()

    # quantas avaliações já tinham prazo até o corte (para a taxa de entrega)
    com_prazo = a.dropna(subset=["date"])
    com_prazo = com_prazo[com_prazo["date"] <= corte]
    devidas = com_prazo.groupby(["code_module", "code_presentation"]).agg(
        aval_n_devidas=("id_assessment", "nunique")).reset_index()
    return out, devidas


def montarDataset(d, corte):
    base = baseEstudantes(d, corte)
    vle = featuresVLE(d, corte)
    aval, devidas = featuresAvaliacoes(d, corte)

    df = base.merge(vle, on=CHAVE, how="left").merge(aval, on=CHAVE, how="left")
    df = df.merge(devidas, on=["code_module", "code_presentation"], how="left")

    # quem nunca apareceu no VLE / nunca entregou nada: zero, não ausente
    for c in df.columns:
        if c.startswith("vle_clicks") or c in ("vle_total_clicks", "vle_dias_ativos", "vle_sites_distintos",
                                                "aval_n_entregues", "aval_n_atrasadas", "aval_n_banked",
                                                "aval_score_ponderado_soma"):
            df[c] = df[c].fillna(0)
    df["aval_n_devidas"] = df["aval_n_devidas"].fillna(0)
    df["aval_taxa_entrega"] = np.where(df["aval_n_devidas"] > 0,
                                       (df["aval_n_entregues"] / df["aval_n_devidas"]).clip(upper=1), np.nan)
    df["nunca_acessou_vle"] = (df["vle_total_clicks"] == 0).astype(int)

    # date_unregistration vaza o alvo: sai do dataset
    df = df.drop(columns=["date_unregistration"])
    return traduzir(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="data/raw", help="pasta com os 7 CSVs originais")
    ap.add_argument("--corte", type=int, default=90, help="dia de corte do dataset principal")
    ap.add_argument("--saida", default="data/oulad_evasao.csv")
    ap.add_argument("--cortes", nargs="*", type=int, help="gera também um arquivo por corte em data/cortes/")
    args = ap.parse_args()

    print("Carregando os CSVs...")
    d = carregarBrutos(args.dados)

    print(f"\nCorte principal: {args.corte} dias")
    df = montarDataset(d, args.corte)
    df.to_csv(args.saida, index=False)
    print(f"Salvo: {args.saida} {df.shape} | taxa de evasão: {df['evasao'].mean():.1%}")

    for corte in (args.cortes or []):
        print(f"\nCorte {corte} dias")
        df = montarDataset(d, corte)
        caminho = Path("data/cortes") / f"oulad_corte_{corte}.csv"
        df.to_csv(caminho, index=False)
        print(f"Salvo: {caminho} {df.shape} | taxa de evasão: {df['evasao'].mean():.1%}")
