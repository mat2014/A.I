"""
gerar_dados.py
Geração da base de dados simulada do Centro Inteligente de Operações
de Mobilidade Urbana.

Lógica geral:
1. Cada registro nasce de um CENÁRIO (normal, pico, chuva, falha
   semafórica, acidente, evento de grande porte), que define faixas
   realistas para as variáveis.
2. Ruído gaussiano é aplicado para evitar dados "perfeitos".
3. O rótulo (status_operacional) é atribuído por REGRAS baseadas nas
   variáveis (e não no cenário), simulando como um operador real
   classificaria a via.
4. ~4% dos rótulos recebem ruído (casos ambíguos de fronteira) e ~2%
   dos valores numéricos viram ausentes, para exigir tratamento no
   pipeline e evitar acurácia artificial de 100%.

Executar:  python src/gerar_dados.py
Saída:     dados/base_original.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

DIAS_UTEIS = ["segunda", "terca", "quarta", "quinta", "sexta"]
FIM_DE_SEMANA = ["sabado", "domingo"]
TODOS_DIAS = DIAS_UTEIS + FIM_DE_SEMANA

TOTAL_REGISTROS = 1500

# Proporção de cada cenário na base
CENARIOS = {
    "normal":  0.40,
    "pico":    0.20,
    "chuva":   0.12,
    "falha":   0.10,
    "acidente": 0.10,
    "evento":  0.08,
}


# ----------------------------------------------------------------------
# 1) Registro base: operação tranquila. Os cenários sobrescrevem campos.
# ----------------------------------------------------------------------
def registros_base(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "hora": rng.integers(0, 24, n),
        "dia_semana": rng.choice(TODOS_DIAS, n),
        "volume_veiculos_h": rng.normal(1800, 450, n),
        "velocidade_media_kmh": rng.normal(60, 9, n),
        "ocupacao_via_pct": rng.normal(35, 12, n),
        "tempo_medio_viagem_min": rng.normal(18, 5, n),
        "atraso_onibus_min": np.abs(rng.normal(3, 2.5, n)),
        "intervalo_onibus_min": rng.normal(8, 2, n),
        "chuva_mm_h": np.zeros(n),
        "visibilidade_m": rng.normal(8500, 900, n),
        "acidentes_reportados": np.zeros(n, dtype=int),
        "falha_semaforica": np.zeros(n, dtype=int),
        "obra_na_via": np.zeros(n, dtype=int),
        "evento_grande_porte": np.zeros(n, dtype=int),
        "frenagens_bruscas_h": np.abs(rng.normal(15, 8, n)),
    })


# ----------------------------------------------------------------------
# 2) Cenários: cada função ajusta o registro base para refletir a
#    situação crítica correspondente (relações causais entre variáveis).
# ----------------------------------------------------------------------
def cenario_normal(n):
    df = registros_base(n)
    # chuva fraca ocasional (20% dos casos), sem grande impacto
    com_chuva = rng.random(n) < 0.20
    df.loc[com_chuva, "chuva_mm_h"] = rng.uniform(1, 8, com_chuva.sum())
    return df


def cenario_pico(n):
    """Horário de pico: volume alto -> velocidade cai, ocupação e tempo sobem."""
    df = registros_base(n)
    df["hora"] = rng.choice([7, 8, 9, 17, 18, 19], n)
    df["dia_semana"] = rng.choice(DIAS_UTEIS, n)
    df["volume_veiculos_h"] = rng.normal(5000, 500, n)
    df["ocupacao_via_pct"] = rng.normal(85, 8, n)
    # velocidade inversamente proporcional à ocupação (+ ruído)
    df["velocidade_media_kmh"] = 90 - 0.85 * df["ocupacao_via_pct"] + rng.normal(0, 4, n)
    df["tempo_medio_viagem_min"] = rng.normal(55, 15, n)
    df["atraso_onibus_min"] = np.abs(rng.normal(12, 6, n))
    df["intervalo_onibus_min"] = rng.normal(14, 4, n)
    df["frenagens_bruscas_h"] = np.abs(rng.normal(60, 25, n))
    return df


def cenario_chuva(n):
    """Chuva intensa: visibilidade cai, velocidade cai, frenagens sobem."""
    df = registros_base(n)
    df["chuva_mm_h"] = rng.uniform(35, 80, n)
    # visibilidade cai com a intensidade da chuva
    df["visibilidade_m"] = np.maximum(80, 4000 - 45 * df["chuva_mm_h"] + rng.normal(0, 300, n))
    df["velocidade_media_kmh"] = rng.normal(32, 7, n)
    df["ocupacao_via_pct"] = rng.normal(65, 12, n)
    df["tempo_medio_viagem_min"] = rng.normal(40, 10, n)
    df["frenagens_bruscas_h"] = np.abs(rng.normal(90, 30, n))
    df["atraso_onibus_min"] = np.abs(rng.normal(10, 5, n))
    return df


def cenario_falha(n):
    """Falha semafórica: retenção local -> ocupação e tempo de viagem sobem."""
    df = registros_base(n)
    df["falha_semaforica"] = 1
    df["velocidade_media_kmh"] = rng.normal(18, 6, n)
    df["ocupacao_via_pct"] = rng.normal(88, 7, n)
    df["tempo_medio_viagem_min"] = rng.normal(65, 15, n)
    df["frenagens_bruscas_h"] = np.abs(rng.normal(110, 35, n))
    df["atraso_onibus_min"] = np.abs(rng.normal(18, 8, n))
    df["volume_veiculos_h"] = rng.normal(3200, 700, n)
    return df


def cenario_acidente(n):
    """Acidente na via: frenagens disparam, velocidade despenca."""
    df = registros_base(n)
    df["acidentes_reportados"] = rng.integers(1, 4, n)
    df["frenagens_bruscas_h"] = np.abs(rng.normal(190, 50, n))
    df["velocidade_media_kmh"] = rng.normal(12, 5, n)
    df["ocupacao_via_pct"] = rng.normal(92, 5, n)
    df["tempo_medio_viagem_min"] = rng.normal(80, 20, n)
    df["atraso_onibus_min"] = np.abs(rng.normal(22, 10, n))
    # 30% dos acidentes ocorrem com chuva (correlação realista)
    com_chuva = rng.random(n) < 0.30
    df.loc[com_chuva, "chuva_mm_h"] = rng.uniform(20, 60, com_chuva.sum())
    return df


def cenario_evento(n):
    """Evento de grande porte: impacto forte no transporte público."""
    df = registros_base(n)
    df["evento_grande_porte"] = 1
    df["hora"] = rng.choice([18, 19, 20, 21, 22], n)
    df["volume_veiculos_h"] = rng.normal(4600, 600, n)
    df["velocidade_media_kmh"] = rng.normal(22, 7, n)
    df["ocupacao_via_pct"] = rng.normal(82, 9, n)
    df["atraso_onibus_min"] = np.abs(rng.normal(32, 12, n))
    df["intervalo_onibus_min"] = rng.normal(25, 7, n)
    df["tempo_medio_viagem_min"] = rng.normal(58, 15, n)
    df["frenagens_bruscas_h"] = np.abs(rng.normal(70, 25, n))
    return df


GERADORES = {
    "normal": cenario_normal,
    "pico": cenario_pico,
    "chuva": cenario_chuva,
    "falha": cenario_falha,
    "acidente": cenario_acidente,
    "evento": cenario_evento,
}


# ----------------------------------------------------------------------
# 3) Limites operacionais do enunciado (clip nas faixas esperadas)
# ----------------------------------------------------------------------
LIMITES = {
    "volume_veiculos_h": (0, 6000),
    "velocidade_media_kmh": (0, 90),
    "ocupacao_via_pct": (0, 100),
    "tempo_medio_viagem_min": (1, 120),
    "atraso_onibus_min": (0, 60),
    "intervalo_onibus_min": (1, 40),
    "chuva_mm_h": (0, 80),
    "visibilidade_m": (50, 10000),
    "acidentes_reportados": (0, 10),
    "frenagens_bruscas_h": (0, 300),
}


def aplicar_limites(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in LIMITES.items():
        df[col] = df[col].clip(lo, hi)
    return df


# ----------------------------------------------------------------------
# 4) Rotulagem por regras (simula o critério de um operador humano).
#    Baseia-se nas VARIÁVEIS, não no cenário de origem.
# ----------------------------------------------------------------------
def classificar(row) -> str:
    # Incidente: condição anormal explícita ou comportamento atípico forte
    if (row["acidentes_reportados"] >= 1
            or row["falha_semaforica"] == 1
            or row["frenagens_bruscas_h"] > 160
            or (row["evento_grande_porte"] == 1 and row["atraso_onibus_min"] > 25)):
        return "INCIDENTE OPERACIONAL"

    # Congestionamento crítico: via saturada e muito lenta
    if row["velocidade_media_kmh"] < 18 and row["ocupacao_via_pct"] > 80:
        return "CONGESTIONAMENTO CRÍTICO"
    if row["tempo_medio_viagem_min"] > 60 and row["ocupacao_via_pct"] > 75:
        return "CONGESTIONAMENTO CRÍTICO"

    # Lentidão moderada: velocidade reduzida ou ocupação elevada
    if row["velocidade_media_kmh"] < 40 or row["ocupacao_via_pct"] > 60:
        return "LENTIDÃO MODERADA"

    return "FLUXO NORMAL"


ORDEM_CLASSES = ["FLUXO NORMAL", "LENTIDÃO MODERADA",
                 "CONGESTIONAMENTO CRÍTICO", "INCIDENTE OPERACIONAL"]


def ruido_de_rotulo(labels: pd.Series, taxa: float = 0.04) -> pd.Series:
    """Troca ~4% dos rótulos por uma classe VIZINHA, simulando casos de
    fronteira em que operadores humanos discordariam. Evita que o modelo
    atinja 100% de acurácia (indício de vazamento, item IX do enunciado)."""
    labels = labels.copy()
    n_trocas = int(len(labels) * taxa)
    idx = rng.choice(labels.index, n_trocas, replace=False)
    for i in idx:
        pos = ORDEM_CLASSES.index(labels[i])
        vizinhos = [p for p in (pos - 1, pos + 1) if 0 <= p < len(ORDEM_CLASSES)]
        labels[i] = ORDEM_CLASSES[rng.choice(vizinhos)]
    return labels


# ----------------------------------------------------------------------
# 5) Valores ausentes propositais (~2% das células numéricas contínuas)
# ----------------------------------------------------------------------
COLS_COM_AUSENTES = ["velocidade_media_kmh", "ocupacao_via_pct",
                     "visibilidade_m", "atraso_onibus_min", "chuva_mm_h"]


def inserir_ausentes(df: pd.DataFrame, taxa: float = 0.02) -> pd.DataFrame:
    for col in COLS_COM_AUSENTES:
        mask = rng.random(len(df)) < taxa
        df.loc[mask, col] = np.nan
    return df


# ----------------------------------------------------------------------
# 6) Montagem final
# ----------------------------------------------------------------------
def main():
    partes = []
    for nome, prop in CENARIOS.items():
        n = int(TOTAL_REGISTROS * prop)
        parte = GERADORES[nome](n)
        parte["cenario_origem"] = nome  # auditoria; remover antes de treinar
        partes.append(parte)

    df = pd.concat(partes, ignore_index=True)
    df = aplicar_limites(df)

    # arredondamentos para parecer telemetria real
    df["volume_veiculos_h"] = df["volume_veiculos_h"].round(0).astype(int)
    for col in ["velocidade_media_kmh", "ocupacao_via_pct", "tempo_medio_viagem_min",
                "atraso_onibus_min", "intervalo_onibus_min", "chuva_mm_h",
                "frenagens_bruscas_h"]:
        df[col] = df[col].round(1)
    df["visibilidade_m"] = df["visibilidade_m"].round(0)

    # rotulagem por regras + ruído de fronteira
    df["status_operacional"] = df.apply(classificar, axis=1)
    df["status_operacional"] = ruido_de_rotulo(df["status_operacional"])

    # valores ausentes propositais
    df = inserir_ausentes(df)

    # embaralha os registros
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    Path("dados").mkdir(exist_ok=True)
    df.to_csv("dados/base_original.csv", index=False)

    print(f"Base gerada: {len(df)} registros -> dados/base_original.csv\n")
    print("Distribuição das classes:")
    print(df["status_operacional"].value_counts(), "\n")
    print("Valores ausentes por coluna:")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()