"""
Agregação do OULAD (Open University Learning Analytics Dataset)
Tema: previsão de evasão escolar.

Unidade de análise: 1 linha = 1 estudante em 1 oferta de módulo
                    (id_student, code_module, code_presentation)
Alvo (target):      evasao = 1 se final_result == 'Withdrawn', senão 0

Uso:
    python agregacao_oulad.py --dados ./oulad --cutoff 90 --saida dataset_evasao.csv [--traduzir]

--cutoff N  -> usa apenas informação disponível até o dia N do curso
               (previsão precoce, sem vazamento de dados).
--cutoff -1 -> usa o curso inteiro (ATENÇÃO: gera vazamento, só para comparação).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]


def carregar(pasta: Path) -> dict:
    d = {}
    d["courses"] = pd.read_csv(pasta / "courses.csv")
    d["assessments"] = pd.read_csv(pasta / "assessments.csv")
    d["student_assessment"] = pd.read_csv(pasta / "studentAssessment.csv")
    d["student_info"] = pd.read_csv(pasta / "studentInfo.csv")
    d["student_reg"] = pd.read_csv(pasta / "studentRegistration.csv")
    d["vle"] = pd.read_csv(pasta / "vle.csv")
    # tabela grande (~10,6M linhas): tipos menores para economizar memória
    d["student_vle"] = pd.read_csv(
        pasta / "studentVle.csv",
        dtype={
            "code_module": "category",
            "code_presentation": "category",
            "id_student": "int32",
            "id_site": "int32",
            "date": "int16",
            "sum_click": "int16",
        },
    )
    return d


# ----------------------------------------------------------------------------
# 1) Base: demografia + registro + alvo
# ----------------------------------------------------------------------------
def base_estudante(d: dict, cutoff: int | None) -> pd.DataFrame:
    base = d["student_info"].merge(d["student_reg"], on=KEY, how="left")
    base = base.merge(d["courses"], on=["code_module", "code_presentation"], how="left")

    base["evasao"] = (base["final_result"] == "Withdrawn").astype(int)

    if cutoff is not None:
        # Quem já cancelou a matrícula até o cutoff já "evadiu": não há o que prever.
        ja_saiu = base["date_unregistration"].notna() & (base["date_unregistration"] <= cutoff)
        print(f"Removendo {ja_saiu.sum()} estudantes que já cancelaram até o dia {cutoff}")
        base = base[~ja_saiu].copy()

    # Atenção: date_unregistration e final_result são vazamento -> não viram feature.
    # date_registration é conhecido antes do início do curso (valor negativo = dias antes).
    base["registrou_tarde"] = (base["date_registration"] > 0).astype(float)
    base.loc[base["date_registration"].isna(), "registrou_tarde"] = np.nan
    return base


# ----------------------------------------------------------------------------
# 2) Features de engajamento no VLE (cliques)
# ----------------------------------------------------------------------------
def features_vle(d: dict, cutoff: int | None) -> pd.DataFrame:
    sv = d["student_vle"]
    if cutoff is not None:
        sv = sv[sv["date"] <= cutoff]

    # junta o tipo de atividade (forumng, quiz, resource, ...)
    vle = d["vle"][["id_site", "code_module", "code_presentation", "activity_type"]]
    sv = sv.merge(vle, on=["id_site", "code_module", "code_presentation"], how="left")
    sv["code_module"] = sv["code_module"].astype(str)
    sv["code_presentation"] = sv["code_presentation"].astype(str)

    g = sv.groupby(KEY, observed=True)
    out = g.agg(
        vle_total_clicks=("sum_click", "sum"),
        vle_dias_ativos=("date", "nunique"),
        vle_sites_distintos=("id_site", "nunique"),
        vle_primeiro_acesso=("date", "min"),
        vle_ultimo_acesso=("date", "max"),
    )
    out["vle_media_clicks_por_dia_ativo"] = out["vle_total_clicks"] / out["vle_dias_ativos"]

    # cliques antes do início do curso (date < 0) e na janela mais recente
    pre = sv[sv["date"] < 0].groupby(KEY, observed=True)["sum_click"].sum().rename("vle_clicks_pre_curso")
    out = out.join(pre)

    if cutoff is not None:
        for janela in (14, 30):
            rec = sv[sv["date"] > cutoff - janela].groupby(KEY, observed=True)["sum_click"].sum()
            out = out.join(rec.rename(f"vle_clicks_ult_{janela}d"))
            out[f"vle_clicks_ult_{janela}d"] = out[f"vle_clicks_ult_{janela}d"].fillna(0)
        out["vle_dias_sem_acesso"] = cutoff - out["vle_ultimo_acesso"]
        out["vle_tendencia_14d"] = out["vle_clicks_ult_14d"] / (out["vle_total_clicks"] + 1)

    # cliques por tipo de atividade (pivot)
    por_tipo = (
        sv.groupby(KEY + ["activity_type"], observed=True)["sum_click"].sum().unstack(fill_value=0)
    )
    por_tipo.columns = [f"vle_clicks_{c}" for c in por_tipo.columns]
    out = out.join(por_tipo)

    out = out.reset_index()
    cols_zero = [c for c in out.columns if c.startswith("vle_clicks") or c in ("vle_total_clicks",)]
    out[cols_zero] = out[cols_zero].fillna(0)
    return out


# ----------------------------------------------------------------------------
# 3) Features de avaliações
# ----------------------------------------------------------------------------
def features_avaliacoes(d: dict, cutoff: int | None) -> pd.DataFrame:
    a = d["assessments"]
    sa = d["student_assessment"].merge(a, on="id_assessment", how="left")
    # a coluna 'date' das avaliações é o prazo (NaN em alguns exames finais)

    if cutoff is not None:
        sa = sa[sa["date_submitted"] <= cutoff]

    sa["atraso_dias"] = sa["date_submitted"] - sa["date"]
    sa["entregue_atrasada"] = (sa["atraso_dias"] > 0).astype(int)
    sa["score_ponderado"] = sa["score"] * sa["weight"] / 100.0

    out = sa.groupby(KEY).agg(
        aval_n_entregues=("id_assessment", "nunique"),
        aval_score_medio=("score", "mean"),
        aval_score_min=("score", "min"),
        aval_score_ponderado_soma=("score_ponderado", "sum"),
        aval_n_atrasadas=("entregue_atrasada", "sum"),
        aval_atraso_medio=("atraso_dias", "mean"),
        aval_n_banked=("is_banked", "sum"),
    ).reset_index()

    # quantas avaliações já tinham prazo até o cutoff (para calcular taxa de entrega)
    av_prazo = a.dropna(subset=["date"])
    if cutoff is not None:
        av_prazo = av_prazo[av_prazo["date"] <= cutoff]
    devidas = av_prazo.groupby(["code_module", "code_presentation"]).agg(
        aval_n_devidas=("id_assessment", "nunique")
    ).reset_index()
    return out, devidas


# ----------------------------------------------------------------------------
# 4) Montagem final
# ----------------------------------------------------------------------------
def montar(pasta: Path, cutoff: int | None, d: dict | None = None) -> pd.DataFrame:
    d = d if d is not None else carregar(pasta)

    base = base_estudante(d, cutoff)
    vle = features_vle(d, cutoff)
    aval, devidas = features_avaliacoes(d, cutoff)

    df = base.merge(vle, on=KEY, how="left").merge(aval, on=KEY, how="left")
    df = df.merge(devidas, on=["code_module", "code_presentation"], how="left")

    # Quem nunca apareceu no VLE / nunca entregou nada: zero, não ausente
    for c in df.columns:
        if c.startswith("vle_clicks") or c in (
            "vle_total_clicks", "vle_dias_ativos", "vle_sites_distintos",
            "aval_n_entregues", "aval_n_atrasadas", "aval_n_banked",
            "aval_score_ponderado_soma",
        ):
            df[c] = df[c].fillna(0)
    df["aval_n_devidas"] = df["aval_n_devidas"].fillna(0)
    df["aval_taxa_entrega"] = np.where(
        df["aval_n_devidas"] > 0,
        (df["aval_n_entregues"] / df["aval_n_devidas"]).clip(upper=1),
        np.nan,
    )
    df["nunca_acessou_vle"] = (df["vle_total_clicks"] == 0).astype(int)

    # Remove colunas que vazam o alvo
    df = df.drop(columns=["date_unregistration"])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="data/raw", help="pasta com os CSVs")
    ap.add_argument("--cutoff", type=int, default=90, help="dia de corte (-1 = curso inteiro)")
    ap.add_argument("--saida", default="data/processed/dataset_evasao_pt.csv")
    ap.add_argument("--traduzir", action="store_true",
                    help="traduz colunas e categorias para português (usa traducao_oulad.py)")
    args = ap.parse_args()

    cutoff = None if args.cutoff < 0 else args.cutoff
    df = montar(Path(args.dados), cutoff)
    if args.traduzir:
        from src.traducao import traduzir
        df = traduzir(df)

    print(f"\nShape final: {df.shape}")
    print(f"Taxa de evasão: {df['evasao'].mean():.1%}")
    print(f"Nulos por coluna (top 8):\n{df.isna().sum().sort_values(ascending=False).head(8)}")
    df.to_csv(args.saida, index=False)
    print(f"\nSalvo em {args.saida}")
