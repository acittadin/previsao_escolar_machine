import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src import config as C

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Cores padrão do projeto
COR_PERMANECE = "#4C78A8"
COR_EVADE = "#E45756"
LABEL = ["permanece", "evade"]

def main():
    print("Iniciando Análise Exploratória de Dados (EDA)...")
    df = pd.read_csv(C.DATASET_PT)

    # 1. Visão Geral do Dataset
    print(f"Total de registros: {df.shape[0]} | Estudantes únicos: {df[C.COL_ID].nunique()}")
    print(f"Taxa média de evasão: {df[C.ALVO].mean():.1%}")
    
    resumo_colunas = pd.DataFrame({
        "tipo": df.dtypes,
        "nulos_qtd": df.isna().sum(),
        "nulos_pct": (df.isna().mean() * 100).round(2),
        "unicos": df.nunique(),
    })
    resumo_colunas.to_csv(C.TABELAS / "eda_resumo_colunas.csv")

    # Gráfico 1: Resultado Final
    fig, ax = plt.subplots(figsize=(6, 4))
    ordem_resultado = ["Aprovado", "Aprovado com distinção", "Reprovado", "Evadido"]
    sns.countplot(data=df, x="resultado_final", order=ordem_resultado, ax=ax, color=COR_PERMANECE)
    
    # Destacando a barra de evasão em vermelho
    if len(ax.patches) > 3:
        ax.patches[3].set_color(COR_EVADE)
        
    ax.set_title("Resultado final dos estudantes", pad=15)
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(C.FIG_EDA / "01_resultado_final.png", dpi=130)
    plt.close()

    # Gráfico 2: Nulos
    nulos = (df.isna().mean() * 100)
    nulos = nulos[nulos > 0].sort_values()
    if not nulos.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        nulos.plot.barh(ax=ax, color="#F58518")
        ax.set_title("Percentual de Valores Ausentes")
        ax.set_xlabel("%")
        plt.tight_layout()
        plt.savefig(C.FIG_EDA / "02_nulos.png", dpi=130)
        plt.close()

    # 2. Análise de Variáveis Categóricas x Evasão
    colunas_categoricas = [
        "escolaridade_max", "faixa_imd", "faixa_etaria", "genero", 
        "deficiencia", "num_tentativas_anteriores", "codigo_modulo", "regiao"
    ]
    
    tabelas_cat = []
    
    for i, col in enumerate(colunas_categoricas, start=3):
        taxa = df.groupby(col)[C.ALVO].agg(['mean', 'count']).rename(columns={'mean': 'taxa_evasao', 'count': 'n'})
        taxa = taxa.sort_values("taxa_evasao", ascending=False)
        
        # Salvando os dados para a tabela final
        tabela_temp = taxa.copy()
        tabela_temp["variavel"] = col
        tabelas_cat.append(tabela_temp)

        # Plotando o gráfico
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(x=taxa.index.astype(str), y=taxa["taxa_evasao"] * 100, color=COR_EVADE, ax=ax)
        ax.axhline(df[C.ALVO].mean() * 100, color="k", ls="--", lw=1, label="Média Geral")
        
        # Colocando o 'n' em cima de cada barra
        for idx, n_val in enumerate(taxa["n"]):
            ax.text(idx, taxa["taxa_evasao"].iloc[idx] * 100 + 0.5, f"n={n_val}", ha="center", fontsize=8)
            
        ax.set_title(f"Taxa de Evasão por {col.replace('_', ' ').title()}")
        ax.set_ylabel("Evasão (%)")
        ax.set_xlabel("")
        ax.legend()
        
        rotacao = 30 if col in ["escolaridade_max", "regiao", "faixa_imd"] else 0
        plt.xticks(rotation=rotacao, ha="right" if rotacao else "center")
        plt.tight_layout()
        plt.savefig(C.FIG_EDA / f"{i:02d}_taxa_{col}.png", dpi=130)
        plt.close()

    # Consolida tabela de categóricas
    pd.concat(tabelas_cat).to_csv(C.TABELAS / "eda_taxa_evasao_categoricas.csv")

    # 3. Análise de Variáveis Numéricas (Desempenho e Engajamento)
    cols_num = [
        "vle_total_cliques", "vle_dias_ativos", "vle_cliques_ult_14d", "vle_dias_sem_acesso",
        "aval_n_entregues", "aval_score_medio", "aval_taxa_entrega", "creditos_cursados"
    ]
    
    comparativo_num = df.groupby(C.ALVO)[cols_num].agg(["mean", "median"]).T.round(2)
    comparativo_num.to_csv(C.TABELAS / "eda_numericas_por_classe.csv")

    # Distribuições (KDE)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.ravel(), cols_num):
        dados_plot = df[[col, C.ALVO]].dropna()
        # Cortar o top 1% para não distorcer o gráfico com outliers extremos
        limite = dados_plot[col].quantile(0.99)
        dados_plot = dados_plot[dados_plot[col] <= limite]
        
        sns.kdeplot(data=dados_plot[dados_plot[C.ALVO] == 0], x=col, ax=ax, color=COR_PERMANECE, fill=True, label=LABEL[0], alpha=0.3)
        sns.kdeplot(data=dados_plot[dados_plot[C.ALVO] == 1], x=col, ax=ax, color=COR_EVADE, fill=True, label=LABEL[1], alpha=0.3)
        ax.set_title(col)
        ax.set_xlabel("")
        
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(C.FIG_EDA / "12_distribuicoes_numericas.png")
    plt.close()

    # 4. Decis de Engajamento
    df["decil_cliques"] = pd.qcut(df["vle_total_cliques"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    evasao_decil = df.groupby("decil_cliques", observed=True)[C.ALVO].mean() * 100
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(evasao_decil.index.astype(str), evasao_decil.values, color=COR_EVADE)
    ax.set_title("Evasão por Decil de Engajamento no VLE")
    ax.set_xlabel("Decil (D1 = Menos cliques)")
    ax.set_ylabel("Evasão (%)")
    plt.tight_layout()
    plt.savefig(C.FIG_EDA / "13_evasao_por_decil_cliques.png", dpi=130)
    plt.close()
    
    evasao_decil.round(1).to_csv(C.TABELAS / "eda_evasao_por_decil_cliques.csv")

    # 5. Matriz de Correlação (Spearman para capturar relações não-lineares)
    print("Calculando correlações...")
    numericas = df.select_dtypes("number").drop(columns=[C.COL_ID], errors="ignore")
    
    corr_alvo = numericas.drop(columns=[C.ALVO]).apply(lambda s: s.corr(df[C.ALVO], method="spearman")).dropna().sort_values()
    corr_alvo.round(3).to_csv(C.TABELAS / "eda_correlacao_com_alvo.csv", header=["spearman"])
    
    fig, ax = plt.subplots(figsize=(8, 9))
    cores_barras = [COR_EVADE if v > 0 else COR_PERMANECE for v in corr_alvo]
    corr_alvo.plot.barh(ax=ax, color=cores_barras)
    ax.set_title("Correlação (Spearman) com Evasão")
    plt.tight_layout()
    plt.savefig(C.FIG_EDA / "15_correlacao_alvo.png", dpi=130)
    plt.close()

    cols_heatmap = ["vle_total_cliques", "vle_dias_ativos", "vle_sites_distintos", "vle_cliques_ult_14d",
                    "vle_dias_sem_acesso", "aval_n_entregues", "aval_score_medio", "aval_taxa_entrega",
                    "aval_n_atrasadas", "creditos_cursados", "num_tentativas_anteriores", C.ALVO]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cols_heatmap].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Matriz de Correlação Geral")
    plt.tight_layout()
    plt.savefig(C.FIG_EDA / "16_matriz_correlacao.png", dpi=130)
    plt.close()

    print("EDA finalizada! Gráficos e tabelas gerados nas pastas de 'reports'.")

if __name__ == "__main__":
    main()