from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


def avaliarModelo(nome, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    acuracia = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # escore contínuo: probabilidade (árvores) ou distância até a margem (SVM)
    escore = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(X_test)
    auc = roc_auc_score(y_test, escore)

    print(f"\n=== {nome} ===")
    print(f"Acuracia: {acuracia:.4f}  (baseline, sempre prever 'permanece': {baseline:.4f})")
    print(f"Recall da evasao: {recall:.4f}  (dos que evadem, quantos o modelo encontra)")
    print(f"ROC-AUC: {auc:.4f}")

    return acuracia, recall, auc
