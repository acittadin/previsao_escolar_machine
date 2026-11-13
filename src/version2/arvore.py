import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from avaliacao import compararParametros
from preprocessamento import criarPreparo


def criarModeloArvore(profundidade=10, folha=100):
    # Árvore não usa distância, então não precisa padronizar (escalar=False).
    # profundidade e folha mínima controlam o sobreajuste: uma árvore profunda decora o treino.
    return Pipeline(steps=[
        ("prep", criarPreparo(escalar=False)),
        ("arvore", DecisionTreeClassifier(max_depth=profundidade, min_samples_leaf=folha,
                                          class_weight="balanced", random_state=42))
    ])


def treinarArvore(X_train, y_train, profundidade=10, folha=100):
    model = criarModeloArvore(profundidade, folha)
    model.fit(X_train, y_train)

    return model


def compararArvore(X_train, y_train, grupos, profundidades=(3, 4, 5, 7, 10), folhas=(20, 50, 100)):
    melhores = compararParametros(criarModeloArvore, {"profundidade": profundidades, "folha": folhas},
                                  X_train, y_train, grupos)
    return melhores["profundidade"], melhores["folha"]


# ----------------------------------------------------------------------------------------------------
# INTERPRETAÇÃO (a vantagem da árvore: dá para ler as regras)
# ----------------------------------------------------------------------------------------------------

def mostrarRegras(model, profundidade=3):
    nomes = model.named_steps["prep"].get_feature_names_out()
    regras = export_text(model.named_steps["arvore"], feature_names=list(nomes), max_depth=profundidade)

    print(f"\n--- Regras da arvore (ate o nivel {profundidade}; class 0 = permanece, 1 = evade) ---")
    print(regras)

    return regras


def mostrarImportancias(model, quantidade=10):
    nomes = model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model.named_steps["arvore"].feature_importances_, index=nomes).sort_values(ascending=False)

    print(f"\n--- {quantidade} variaveis mais usadas pela arvore ---")
    print(imp.head(quantidade).round(4))

    return imp
