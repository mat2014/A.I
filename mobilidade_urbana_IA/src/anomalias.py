"""
anomalias.py
Camada de detecção de anomalias do Centro Inteligente de Operações
de Mobilidade Urbana, usando Isolation Forest.

Papel na arquitetura:
- O CLASSIFICADOR responde "em qual das 4 classes conhecidas a via está?"
- O DETECTOR DE ANOMALIAS responde "esse registro é um padrão RARO,
  diferente do comportamento habitual da via?"

Eles se complementam: uma combinação inédita de variáveis (ex.: chuva
forte de madrugada com volume alto) pode ser classificada com baixa
confiança pelo modelo — o detector sinaliza que o registro foge do
padrão e merece atenção humana, funcionando como rede de segurança.

Como o Isolation Forest funciona (intuição):
- Constrói árvores que particionam os dados com cortes aleatórios.
- Pontos "normais" (em regiões densas) precisam de MUITOS cortes para
  serem isolados; pontos anômalos (isolados no espaço) são separados
  com POUCOS cortes.
- O score de anomalia mede quão fácil foi isolar o ponto.

Executar (da raiz do projeto):  python src/anomalias.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42
CONTAMINACAO = 0.05  # esperamos ~5% de registros anômalos na operação


# ----------------------------------------------------------------------
# Pré-processamento (mesmas regras do treinar.py)
# ----------------------------------------------------------------------
def preprocessar(caminho_csv: str):
    df = pd.read_csv(caminho_csv)
    df_original = df.copy()  # guardamos para inspecionar as anomalias

    if "cenario_origem" in df.columns:
        df = df.drop(columns=["cenario_origem"])

    for col in df.columns[df.isna().any()]:
        df[col] = df[col].fillna(df[col].median())

    df = pd.get_dummies(df, columns=["dia_semana"], dtype=int)

    X = df.drop(columns=["status_operacional"])
    y = df["status_operacional"]
    return X, y, df_original


def main():
    Path("modelos").mkdir(exist_ok=True)
    Path("resultados").mkdir(exist_ok=True)

    X, y, df_original = preprocessar("dados/base_original.csv")

    # usa o MESMO scaler do treinamento (consistência com a interface)
    scaler = joblib.load("modelos/scaler.pkl")
    X_s = scaler.transform(X)

    # ------------------------------------------------------------------
    # treina o Isolation Forest
    # ------------------------------------------------------------------
    detector = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINACAO,
        random_state=RANDOM_STATE,
    )
    detector.fit(X_s)

    # -1 = anômalo | 1 = normal
    flags = detector.predict(X_s)
    scores = detector.decision_function(X_s)  # quanto menor, mais anômalo

    df_original["anomalia"] = (flags == -1)
    df_original["score_anomalia"] = scores.round(3)

    n_anomalos = int(df_original["anomalia"].sum())
    print(f"Registros marcados como anômalos: {n_anomalos} de "
          f"{len(df_original)} ({n_anomalos/len(df_original):.1%})\n")

    # ------------------------------------------------------------------
    # onde estão as anomalias? cruzamento com a classe operacional
    # ------------------------------------------------------------------
    cruzamento = pd.crosstab(df_original["status_operacional"],
                             df_original["anomalia"],
                             normalize="index").round(3) * 100
    cruzamento.columns = ["Normal (%)", "Anômalo (%)"]
    print("=== Percentual de anomalias por classe ===")
    print(cruzamento, "\n")

    # gráfico: distribuição do score por classe
    plt.figure(figsize=(11, 5))
    ordem = ["FLUXO NORMAL", "LENTIDÃO MODERADA",
             "CONGESTIONAMENTO CRÍTICO", "INCIDENTE OPERACIONAL"]
    dados_box = [df_original.loc[df_original["status_operacional"] == c,
                                 "score_anomalia"] for c in ordem]
    plt.boxplot(dados_box, tick_labels=[c.replace(" ", "\n") for c in ordem])
    plt.axhline(0, color="red", linestyle="--",
                label="limiar (abaixo = anômalo)")
    plt.ylabel("Score de anomalia")
    plt.title("Score de anomalia por classe operacional (Isolation Forest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultados/anomalias_por_classe.png", dpi=150)
    print("Gráfico salvo em resultados/anomalias_por_classe.png\n")

    # ------------------------------------------------------------------
    # exemplos concretos: os 5 registros mais anômalos
    # ------------------------------------------------------------------
    colunas_interesse = ["hora", "volume_veiculos_h", "velocidade_media_kmh",
                         "chuva_mm_h", "acidentes_reportados",
                         "falha_semaforica", "evento_grande_porte",
                         "frenagens_bruscas_h", "atraso_onibus_min",
                         "status_operacional", "score_anomalia"]
    top5 = df_original.nsmallest(5, "score_anomalia")[colunas_interesse]
    print("=== 5 registros mais anômalos ===")
    print(top5.to_string(index=False), "\n")

    # salva base com flags (útil para o PDF) e o detector para a interface
    df_original.to_csv("resultados/base_com_anomalias.csv", index=False)
    joblib.dump(detector, "modelos/detector_anomalia.pkl")
    print("Detector salvo em modelos/detector_anomalia.pkl")


if __name__ == "__main__":
    main()