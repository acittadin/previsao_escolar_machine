import argparse
import subprocess
import sys
import time

from src import clusterizacao, config as C, eda, experimento_cortes, limpeza, modelagem

ETAPAS_DISPONIVEIS = ["agregacao", "limpeza", "eda", "kmeans", "modelagem", "cortes"]

def main():
    parser = argparse.ArgumentParser(description="Pipeline do Projeto de Evasão Escolar (OULAD)")
    parser.add_argument("--etapas", nargs="+", choices=ETAPAS_DISPONIVEIS, 
                        help="Etapas específicas para rodar (ex: limpeza eda kmeans modelagem)")
    parser.add_argument("--agregar", action="store_true", help="Força a agregação dos CSVs brutos primeiro")
    parser.add_argument("--rapido", action="store_true", help="Roda a modelagem mais rápido (grades menores para teste)")
    
    args = parser.parse_args()

    # Se o usuário não passar etapas específicas, roda o fluxo padrão
    etapas_para_rodar = args.etapas
    if not etapas_para_rodar:
        etapas_para_rodar = ["limpeza", "eda", "kmeans", "modelagem"]
        if args.agregar:
            etapas_para_rodar.insert(0, "agregacao")

    for etapa in etapas_para_rodar:
        print(f"\n{'=' * 60}\n>>> EXECUTANDO ETAPA: {etapa.upper()}\n{'=' * 60}")
        t0 = time.time()
        
        if etapa == "agregacao":
            arquivo_vle = C.DADOS_RAW / "studentVle.csv"
            if not arquivo_vle.exists():
                sys.exit(f"Erro: Arquivos brutos não encontrados em {C.DADOS_RAW}. Coloque os CSVs na pasta primeiro.")
            
            # Executa o script de agregação passando os argumentos via linha de comando
            subprocess.run([
                sys.executable, "-m", "src.agregacao", 
                "--dados", str(C.DADOS_RAW),
                "--cutoff", str(C.CUTOFF_DIAS), 
                "--traduzir", 
                "--saida", str(C.DATASET_PT)
            ], check=True)
            
        elif etapa == "limpeza":
            limpeza.main()
            
        elif etapa == "eda":
            eda.main()
            
        elif etapa == "kmeans":
            clusterizacao.main()
            
        elif etapa == "modelagem":
            modelagem.main(rapido=args.rapido)
            
        elif etapa == "cortes":
            if (C.DADOS_RAW / "studentVle.csv").exists():
                experimento_cortes.agregar_cortes(experimento_cortes.CORTES_PADRAO, C.DADOS_RAW)
            else:
                print("CSVs brutos ausentes. Usando os datasets já gerados na pasta data/processed/cortes/")

            experimento_cortes.modelar(experimento_cortes.CORTES_PADRAO)
            experimento_cortes.relatorio()
            
        tempo_gasto = time.time() - t0
        print(f"[{etapa}] finalizada em {tempo_gasto:.0f} segundos.")

if __name__ == "__main__":
    main()