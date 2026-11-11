import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Chave primária: um estudante em um módulo e uma oferta de semestre específica
CHAVE_PRIMARIA = ["code_module", "code_presentation", "id_student"]

def carregar_dados_brutos(pasta: Path) -> dict:
    """
    Lê os 7 arquivos CSV originais do OULAD e os coloca num dicionário.
    A tabela studentVle (com ~10 milhões de linhas) é carregada com tipos otimizados
    para economizar memória (category e int16).
    """
    print(f"Carregando arquivos da pasta: {pasta}")
    d = {}
    d["courses"] = pd.read_csv(pasta / "courses.csv")
    d["assessments"] = pd.read_csv(pasta / "assessments.csv")
    d["student_assessment"] = pd.read_csv(pasta / "studentAssessment.csv")
    d["student_info"] = pd.read_csv(pasta / "studentInfo.csv")
    d["student_reg"] = pd.read_csv(pasta / "studentRegistration.csv")
    d["vle"] = pd.read_csv(pasta / "vle.csv")
    
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


def preparar_demografia_e_alvo(d: dict, corte_dias: int) -> pd.DataFrame:
    """
    Junta as informações do estudante com os dados do curso e cria a variável alvo 'evadiu'.
    Alunos que já cancelaram a matrícula ANTES do dia de corte são removidos, 
    pois não há sentido em prever a evasão de quem já evadiu.
    """
    df = d["student_info"].merge(d["student_reg"], on=CHAVE_PRIMARIA, how="left")
    df = df.merge(d["courses"], on=["code_module", "code_presentation"], how="left")

    # Criação do Alvo (1 para evasão, 0 para aprovados/reprovados)
    df["evadiu"] = (df["final_result"] == "Withdrawn").astype(int)

    if corte_dias is not None:
        cancelou_cedo = df["date_unregistration"].notna() & (df["date_unregistration"] <= corte_dias)
        print(f"Removendo {cancelou_cedo.sum()} estudantes que já cancelaram até o dia {corte_dias}")
        df = df[~cancelou_cedo].copy()

    # O registro tem valores negativos para alunos que se matricularam dias antes de o curso iniciar
    df["registrou_tarde"] = (df["date_registration"] > 0).astype(float)
    df.loc[df["date_registration"].isna(), "registrou_tarde"] = np.nan
    
    return df


def preparar_engajamento_vle(d: dict, corte_dias: int) -> pd.DataFrame:
    """
    Agrupa os milhões de cliques no ambiente virtual (VLE) por estudante.
    Se um corte temporal for definido, ignora todos os cliques que aconteceram depois do dia de corte.
    """
    cliques = d["student_vle"]
    
    if corte_dias is not None:
        cliques = cliques[cliques["date"] <= corte_dias]

    # Cruza com a tabela VLE para saber o tipo de material clicado (PDF, fórum, quiz, etc)
    tipos_vle = d["vle"][["id_site", "code_module", "code_presentation", "activity_type"]]
    cliques = cliques.merge(tipos_vle, on=["id_site", "code_module", "code_presentation"], how="left")
    
    # Corrige os tipos das colunas categóricas perdidos no merge
    cliques["code_module"] = cliques["code_module"].astype(str)
    cliques["code_presentation"] = cliques["code_presentation"].astype(str)

    agrupado = cliques.groupby(CHAVE_PRIMARIA, observed=True)
    df_engajamento = agrupado.agg(
        vle_total_cliques=("sum_click", "sum"),
        vle_dias_ativos=("date", "nunique"),
        vle_sites_distintos=("id_site", "nunique"),
        vle_primeiro_acesso=("date", "min"),
        vle_ultimo_acesso=("date", "max"),
    )
    df_engajamento["vle_media_cliques_por_dia_ativo"] = df_engajamento["vle_total_cliques"] / df_engajamento["vle_dias_ativos"]

    # Calcula engajamento antes de as aulas começarem (dias negativos)
    cliques_pre = cliques[cliques["date"] < 0].groupby(CHAVE_PRIMARIA, observed=True)["sum_click"].sum().rename("vle_cliques_pre_curso")
    df_engajamento = df_engajamento.join(cliques_pre)

    if corte_dias is not None:
        for janela in (14, 30):
            cliques_recentes = cliques[cliques["date"] > (corte_dias - janela)].groupby(CHAVE_PRIMARIA, observed=True)["sum_click"].sum()
            df_engajamento = df_engajamento.join(cliques_recentes.rename(f"vle_cliques_ult_{janela}d"))
            df_engajamento[f"vle_cliques_ult_{janela}d"] = df_engajamento[f"vle_cliques_ult_{janela}d"].fillna(0)
            
        df_engajamento["vle_dias_sem_acesso"] = corte_dias - df_engajamento["vle_ultimo_acesso"]
        # Tendência: quantos cliques dos últimos 14 dias representam do total acumulado?
        df_engajamento["vle_tendencia_14d"] = df_engajamento["vle_cliques_ult_14d"] / (df_engajamento["vle_total_cliques"] + 1)

    # Transforma o tipo de atividade do VLE em colunas (pivot)
    por_tipo = cliques.groupby(CHAVE_PRIMARIA + ["activity_type"], observed=True)["sum_click"].sum().unstack(fill_value=0)
    por_tipo.columns = [f"vle_cliques_{c}" for c in por_tipo.columns]
    df_engajamento = df_engajamento.join(por_tipo).reset_index()

    # Se a pessoa não tem cliques num tipo específico, preenche com 0
    colunas_com_cliques = [c for c in df_engajamento.columns if c.startswith("vle_cliques") or c == "vle_total_cliques"]
    df_engajamento[colunas_com_cliques] = df_engajamento[colunas_com_cliques].fillna(0)
    
    return df_engajamento


def preparar_avaliacoes(d: dict, corte_dias: int) -> pd.DataFrame:
    """
    Mapeia as notas, entregas e atrasos dos alunos nas avaliações.
    Respeita o corte temporal para não contar notas que o aluno só tirou no fim do curso.
    """
    avaliacoes_gerais = d["assessments"]
    entregas_aluno = d["student_assessment"].merge(avaliacoes_gerais, on="id_assessment", how="left")

    if corte_dias is not None:
        entregas_aluno = entregas_aluno[entregas_aluno["date_submitted"] <= corte_dias]

    entregas_aluno["atraso_dias"] = entregas_aluno["date_submitted"] - entregas_aluno["date"]
    entregas_aluno["entregue_atrasada"] = (entregas_aluno["atraso_dias"] > 0).astype(int)
    entregas_aluno["score_ponderado"] = entregas_aluno["score"] * entregas_aluno["weight"] / 100.0

    df_avaliacoes = entregas_aluno.groupby(CHAVE_PRIMARIA).agg(
        aval_n_entregues=("id_assessment", "nunique"),
        aval_score_medio=("score", "mean"),
        aval_score_min=("score", "min"),
        aval_score_ponderado_soma=("score_ponderado", "sum"),
        aval_n_atrasadas=("entregue_atrasada", "sum"),
        aval_atraso_medio=("atraso_dias", "mean"),
        aval_n_banked=("is_banked", "sum"),
    ).reset_index()

    # Conta quantas provas o aluno DEVERIA ter feito até o dia de corte
    av_prazo = avaliacoes_gerais.dropna(subset=["date"])
    if corte_dias is not None:
        av_prazo = av_prazo[av_prazo["date"] <= corte_dias]
        
    provas_devidas = av_prazo.groupby(["code_module", "code_presentation"]).agg(
        aval_n_devidas=("id_assessment", "nunique")
    ).reset_index()
    
    return df_avaliacoes, provas_devidas


def montar_dataset_final(pasta: Path, corte_dias: int, d: dict | None = None) -> pd.DataFrame:
    """Consolida as três bases agregadas em um único Dataset rico."""
    d = carregar_dados_brutos(pasta) if d is None else d

    print("Consolidando Demografia, Engajamento no VLE e Histórico de Avaliações...")
    df_base = preparar_demografia_e_alvo(d, corte_dias)
    df_vle = preparar_engajamento_vle(d, corte_dias)
    df_aval, df_devidas = preparar_avaliacoes(d, corte_dias)

    df_final = df_base.merge(df_vle, on=CHAVE_PRIMARIA, how="left").merge(df_aval, on=CHAVE_PRIMARIA, how="left")
    df_final = df_final.merge(df_devidas, on=["code_module", "code_presentation"], how="left")

    # Alunos que nunca entraram no portal ou não fizeram provas ganham 0 por padrão, não nulo (NaN)
    colunas_para_zerar = [
        c for c in df_final.columns if c.startswith("vle_cliques") or c in (
            "vle_total_cliques", "vle_dias_ativos", "vle_sites_distintos",
            "aval_n_entregues", "aval_n_atrasadas", "aval_n_banked",
            "aval_score_ponderado_soma"
        )
    ]
    for c in colunas_para_zerar:
        df_final[c] = df_final[c].fillna(0)
        
    df_final["aval_n_devidas"] = df_final["aval_n_devidas"].fillna(0)
    
    # Calcula a taxa de provas entregues (evitando divisão por zero)
    df_final["aval_taxa_entrega"] = np.where(
        df_final["aval_n_devidas"] > 0,
        (df_final["aval_n_entregues"] / df_final["aval_n_devidas"]).clip(upper=1),
        np.nan,
    )
    df_final["nunca_acessou_vle"] = (df_final["vle_total_cliques"] == 0).astype(int)

    # Removendo datas do futuro que causariam Data Leakage
    df_final = df_final.drop(columns=["date_unregistration"])
    
    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", default="data/raw", help="Caminho da pasta com os arquivos CSV originais")
    parser.add_argument("--cutoff", type=int, default=90, help="Dia de corte para simular a evasão no meio do curso")
    parser.add_argument("--saida", default="data/processed/dataset_evasao_pt.csv")
    parser.add_argument("--traduzir", action="store_true", help="Se True, traduz categorias para o português")
    args = parser.parse_args()

    corte_dias = None if args.cutoff < 0 else args.cutoff
    
    dataset_completo = montar_dataset_final(Path(args.dados), corte_dias)
    
    if args.traduzir:
        from src.traducao import traduzir
        dataset_completo = traduzir(dataset_completo)

    print(f"\nResumo da Base de Evasão (Corte: {corte_dias} dias):")
    print(f"Total de Linhas: {dataset_completo.shape[0]} | Colunas: {dataset_completo.shape[1]}")
    print(f"Taxa Média de Evasão: {dataset_completo['evadiu'].mean():.1%}")
    
    dataset_completo.to_csv(args.saida, index=False)
    print(f"\nDataset consolidado com sucesso em: {args.saida}")