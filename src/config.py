"""Configurações centrais do projeto (caminhos, semente, parâmetros do experimento)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Caminhos
DADOS_RAW = RAIZ / "data" / "raw"
DADOS_PROC = RAIZ / "data" / "processed"
DATASET_PT = DADOS_PROC / "dataset_evasao_pt.csv"      # saída da agregação (+ tradução)
DATASET_LIMPO = DADOS_PROC / "dataset_limpo.csv"       # saída da limpeza
FIG_EDA = RAIZ / "reports" / "figures" / "eda"
FIG_CLUSTERS = RAIZ / "reports" / "figures" / "clusters"
FIG_MODELOS = RAIZ / "reports" / "figures" / "modelos"
TABELAS = RAIZ / "reports" / "tables"
MODELOS = RAIZ / "models"

# Experimento
SEED = 42
CUTOFF_DIAS = 90                 # só informação até o dia 90 do curso (previsão precoce)
ALVO = "evasao"
COL_ID = "id_estudante"
COL_OFERTA = "codigo_oferta"
OFERTA_TESTE = "2014J"           # split temporal: treina nas ofertas anteriores, testa na mais recente
CAT_COLS = ["regiao", "codigo_modulo"]   # one-hot na modelagem

for _p in (DADOS_PROC, FIG_EDA, FIG_CLUSTERS, FIG_MODELOS, TABELAS, MODELOS):
    _p.mkdir(parents=True, exist_ok=True)
