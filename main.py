"""
Executa o pipeline completo do projeto.

    python main.py                       # limpeza -> EDA -> KMeans -> modelagem
    python main.py --etapas eda kmeans   # só algumas etapas
    python main.py --agregar             # inclui a agregação a partir dos CSVs brutos (data/raw)
    python main.py --rapido              # modelagem com grades reduzidas (teste rápido)

Etapas: agregacao, limpeza, eda, kmeans, modelagem, cortes
    python main.py --etapas cortes        # experimento com cortes de 0 a 180 dias (~12 min)
"""
import argparse
import subprocess
import sys
import time

from src import clusterizacao, config as C, eda, experimento_cortes, limpeza, modelagem

ETAPAS = ["agregacao", "limpeza", "eda", "kmeans", "modelagem", "cortes"]
OPCIONAIS = {"agregacao", "cortes"}   # só rodam se pedidas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--etapas", nargs="+", choices=ETAPAS, help="etapas a executar (padrão: limpeza, eda, kmeans, modelagem)")
    ap.add_argument("--agregar", action="store_true", help="inclui a etapa de agregação dos dados brutos")
    ap.add_argument("--rapido", action="store_true", help="modelagem com grades reduzidas")
    a = ap.parse_args()

    etapas = a.etapas or [e for e in ETAPAS if e not in OPCIONAIS or (e == "agregacao" and a.agregar)]

    for etapa in etapas:
        print(f"\n{'=' * 78}\n>>> ETAPA: {etapa.upper()}\n{'=' * 78}")
        t0 = time.time()
        if etapa == "agregacao":
            if not (C.DADOS_RAW / "studentVle.csv").exists():
                sys.exit(f"CSVs brutos não encontrados em {C.DADOS_RAW}. Veja o README.")
            subprocess.run([sys.executable, "-m", "src.agregacao", "--dados", str(C.DADOS_RAW),
                            "--cutoff", str(C.CUTOFF_DIAS), "--traduzir", "--saida", str(C.DATASET_PT)],
                           check=True)
        elif etapa == "limpeza":
            limpeza.main()
        elif etapa == "eda":
            eda.main()
        elif etapa == "kmeans":
            clusterizacao.main()
        elif etapa == "modelagem":
            modelagem.main(rapido=a.rapido)
        elif etapa == "cortes":
            E = experimento_cortes
            if (C.DADOS_RAW / "studentVle.csv").exists():
                E.agregar(E.CORTES_PADRAO, C.DADOS_RAW)
            else:
                print("CSVs brutos ausentes: usando os datasets por corte já salvos em data/processed/cortes/")
            E.modelar(E.CORTES_PADRAO)
            E.relatorio()
        print(f"[{etapa}] concluída em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
