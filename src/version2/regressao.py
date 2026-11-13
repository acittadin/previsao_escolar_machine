import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from avaliacao import compararParametros
from preprocessamento import criarPreparo

# A regressão logística é o modelo de REFERÊNCIA (baseline): os outros só valem a pena
# se forem melhores que ela.


def criarModelo(c=1):
    # class_weight="balanced": a evasão é rara (~12%), então os erros na classe 1 pesam mais no ajuste.
    return Pipeline(steps=[
        ("prep", criarPreparo(escalar=True)),
        ("logit", LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=42))
    ])


def treinarRegressao(X_train, y_train, c=1):
    model = criarModelo(c)
    model.fit(X_train, y_train)

    return model


def compararC(X_train, y_train, grupos, valores=(0.01, 0.1, 1)):
    return compararParametros(criarModelo, {"c": valores}, X_train, y_train, grupos)["c"]


# ----------------------------------------------------------------------------------------------------
# INTERPRETAÇÃO
# ----------------------------------------------------------------------------------------------------

def mostrarCoeficientes(model, quantidade=10):
    nomes = model.named_steps["prep"].get_feature_names_out()
    pesos = model.named_steps["logit"].coef_[0]
    coef = pd.Series(pesos, index=nomes).sort_values(ascending=False)

    print("\n--- Coeficientes (classe 1 = evasao; variaveis padronizadas, entao sao comparaveis) ---")
    print(f"Aumentam o risco:\n{coef.head(quantidade).round(4)}")
    print(f"\nReduzem o risco:\n{coef.tail(quantidade).round(4)}")

    return coef
