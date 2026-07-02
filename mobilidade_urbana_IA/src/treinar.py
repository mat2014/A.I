"""
treinar.py
Treinamento, comparação e validação dos modelos de classificação do
Centro Inteligente de Operações de Mobilidade Urbana.

Pipeline:
1. Lê a base original e repete o pré-processamento (script autossuficiente
   e reprodutível, independente do notebook).
2. Split estratificado 70/15/15 (treino/validação/teste).
3. Treina 3 modelos: Árvore de Decisão, Random Forest e MLP.
4. Compara métricas na VALIDAÇÃO e escolhe o melhor por F1-macro.
5. Avalia o modelo final no TESTE (dados nunca vistos).
6. Salva: modelo_final.pkl, scaler.pkl, colunas.pkl, gráficos e métricas.

Executar (da raiz do projeto):  python src/treinar.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # permite salvar gráficos sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 42
ORDEM_CLASSES = ["FLUXO NORMAL", "LENTIDÃO MODERADA",
                 "CONGESTIONAMENTO CRÍTICO", "INCIDENTE OPERACIONAL"]


# ----------------------------------------------------------------------
# 1) Pré-processamento (mesmas regras do notebook de EDA)
# ----------------------------------------------------------------------
def preprocessar(caminho_csv: str):
    df = pd.read_csv(caminho_csv)

    # remove coluna de auditoria da geração (evita vazamento)
    if "cenario_origem" in df.columns:
        df = df.drop(columns=["cenario_origem"])

    # imputação de ausentes pela mediana
    for col in df.columns[df.isna().any()]:
        df[col] = df[col].fillna(df[col].median())

    # one-hot no dia da semana
    df = pd.get_dummies(df, columns=["dia_semana"], dtype=int)

    X = df.drop(columns=["status_operacional"])
    y = df["status_operacional"]
    return X, y


# ----------------------------------------------------------------------
# 2) Treino e comparação
# ----------------------------------------------------------------------
def main():
    Path("modelos").mkdir(exist_ok=True)
    Path("resultados").mkdir(exist_ok=True)

    X, y = preprocessar("dados/base_original.csv")

    # split 70 / 15 / 15 estratificado
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp,
        random_state=RANDOM_STATE)

    print(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}\n")

    # padronização: fit APENAS no treino (evita vazamento de dados)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    modelos = {
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE),
        "MLP (rede neural)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=800,
            random_state=RANDOM_STATE),
    }

    # ------------------------------------------------------------------
    # comparação na VALIDAÇÃO
    # ------------------------------------------------------------------
    resultados = []
    treinados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train_s, y_train)
        pred_val = modelo.predict(X_val_s)
        resultados.append({
            "Modelo": nome,
            "Acurácia": accuracy_score(y_val, pred_val),
            "Precision (macro)": precision_score(y_val, pred_val, average="macro"),
            "Recall (macro)": recall_score(y_val, pred_val, average="macro"),
            "F1 (macro)": f1_score(y_val, pred_val, average="macro"),
        })
        treinados[nome] = modelo

    tabela = pd.DataFrame(resultados).set_index("Modelo").round(4)
    print("=== Comparação na VALIDAÇÃO ===")
    print(tabela, "\n")
    tabela.to_csv("resultados/comparacao_modelos.csv")

    # escolha do melhor por F1-macro (métrica justa para base desbalanceada)
    melhor_nome = tabela["F1 (macro)"].idxmax()
    melhor = treinados[melhor_nome]
    print(f">>> Modelo escolhido: {melhor_nome} (maior F1-macro na validação)\n")

    # ------------------------------------------------------------------
    # avaliação final no TESTE (dados nunca vistos)
    # ------------------------------------------------------------------
    pred_test = melhor.predict(X_test_s)
    print("=== Avaliação final no TESTE ===")
    print(classification_report(y_test, pred_test, digits=3))

    with open("resultados/relatorio_teste.txt", "w", encoding="utf-8") as f:
        f.write(f"Modelo final: {melhor_nome}\n\n")
        f.write(classification_report(y_test, pred_test, digits=3))

    # matriz de confusão
    cm = confusion_matrix(y_test, pred_test, labels=ORDEM_CLASSES)
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(cm, display_labels=ORDEM_CLASSES).plot(
        ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    ax.set_title(f"Matriz de confusão — {melhor_nome} (conjunto de teste)")
    plt.tight_layout()
    plt.savefig("resultados/matriz_confusao.png", dpi=150)
    print("Matriz de confusão salva em resultados/matriz_confusao.png")

    # análise dos principais erros (para o PDF e a arguição)
    erros = pd.DataFrame({"real": y_test.values, "previsto": pred_test})
    erros = erros[erros["real"] != erros["previsto"]]
    print(f"\nErros no teste: {len(erros)} de {len(y_test)} "
          f"({len(erros)/len(y_test):.1%})")
    print("Pares (real -> previsto) mais frequentes:")
    print(erros.value_counts().head(5))

    # importância das variáveis (se o modelo tiver)
    if hasattr(melhor, "feature_importances_"):
        imp = pd.Series(melhor.feature_importances_, index=X.columns)
        imp = imp.sort_values().tail(12)
        plt.figure(figsize=(9, 6))
        imp.plot(kind="barh", color="#1565c0")
        plt.title(f"Importância das variáveis — {melhor_nome}")
        plt.tight_layout()
        plt.savefig("resultados/importancia_variaveis.png", dpi=150)
        print("\nImportância das variáveis salva em "
              "resultados/importancia_variaveis.png")

    # ------------------------------------------------------------------
    # salva artefatos para a interface (app.py)
    # ------------------------------------------------------------------
    joblib.dump(melhor, "modelos/modelo_final.pkl")
    joblib.dump(scaler, "modelos/scaler.pkl")
    joblib.dump(list(X.columns), "modelos/colunas.pkl")
    print("\nArtefatos salvos: modelo_final.pkl, scaler.pkl, colunas.pkl")


if __name__ == "__main__":
    main()