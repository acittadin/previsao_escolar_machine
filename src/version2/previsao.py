import numpy as np
import pandas as pd

from preprocessamento import COLUNAS_CATEGORICAS


def perfilTipico(X_referencia):
    # Perfil "típico" do treino: mediana nas numéricas e valor mais comum nas categóricas.
    # O aluno novo só informa o que quiser; o resto assume esse perfil.
    perfil = {}
    for coluna in X_referencia.columns:
        if coluna in COLUNAS_CATEGORICAS:
            perfil[coluna] = X_referencia[coluna].mode().iloc[0]
        else:
            perfil[coluna] = float(X_referencia[coluna].median())
    return perfil


def preverNovoAluno(modelo_treinado, dados_aluno, limiar, X_referencia):
    if not isinstance(dados_aluno, dict):
        raise TypeError("dados_aluno deve ser um dicionário.")

    perfil = perfilTipico(X_referencia)
    colunas_log = {c.replace("log_", "", 1) for c in X_referencia.columns if c.startswith("log_")}

    desconhecidas = [k for k in dados_aluno if k not in perfil and k not in colunas_log]
    if desconhecidas:
        raise ValueError(f"Chaves desconhecidas em dados_aluno: {desconhecidas}. "
                         f"Colunas validas: {sorted(perfil) + sorted(colunas_log)}")

    for chave, valor in dados_aluno.items():
        if chave in colunas_log:                      # cliques entram em escala original, o modelo usa log(1 + x)
            perfil[f"log_{chave}"] = float(np.log1p(valor))
        elif valor is None:                           # ausente: o pipeline imputa pela mediana do treino
            perfil[chave] = np.nan
        else:
            perfil[chave] = valor

    df_aluno = pd.DataFrame([perfil], columns=list(X_referencia.columns))

    if hasattr(modelo_treinado, "predict_proba"):
        escore = float(modelo_treinado.predict_proba(df_aluno)[0][1])
        # Atenção: com class_weight="balanced" o escore NÃO é uma probabilidade real (ele superestima o risco).
        # Use para ORDENAR alunos e comparar com o limiar, não como "chance de evasão".
        rotulo_escore = f"Escore de risco: {escore:.2%} (serve para ordenar alunos; nao e uma probabilidade calibrada)"
    elif hasattr(modelo_treinado, "decision_function"):
        escore = float(modelo_treinado.decision_function(df_aluno)[0])
        rotulo_escore = f"Escore de risco (distancia a margem do SVM): {escore:+.3f}"
    else:
        raise AttributeError("O modelo treinado não possui predict_proba() nem decision_function().")

    em_risco = escore >= limiar

    print("\n" + "=" * 60)
    print("SISTEMA DE ALERTA: AVALIAÇÃO DE NOVO ALUNO")
    print("=" * 60)
    print(rotulo_escore)
    if em_risco:
        print("ALERTA: aluno classificado como EM RISCO de evasao.")
        print(f"(Limiar de {limiar:.3f} ultrapassado)")
    else:
        print("Aluno classificado como SEM RISCO IMEDIATO de evasao.")
    print("=" * 60 + "\n")

    return {"predicao": "em risco" if em_risco else "sem risco imediato", "escore": escore, "limiar": limiar}
