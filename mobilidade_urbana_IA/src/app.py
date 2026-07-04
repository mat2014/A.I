
# src/app.py
# Interface Streamlit — Centro Inteligente de Operações de Mobilidade Urbana
# Etapa 1: estrutura, carregamento de modelos e entrada de dados

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------- Configuração da página ----------
st.set_page_config(
    page_title="Centro de Operações de Mobilidade",
    page_icon="🚦",
    layout="wide",
)

# ---------- Carregamento dos artefatos (com cache) ----------
BASE = Path(__file__).resolve().parent.parent  # pasta do projeto

@st.cache_resource
def carregar_artefatos():
    modelo = joblib.load(BASE / "modelos" / "modelo_final.pkl")
    scaler = joblib.load(BASE / "modelos" / "scaler.pkl")
    colunas = joblib.load(BASE / "modelos" / "colunas.pkl")
    detector = joblib.load(BASE / "modelos" / "detector_anomalia.pkl")
    return modelo, scaler, colunas, detector

modelo, scaler, colunas, detector = carregar_artefatos()

# ---------- Título ----------
st.title("🚦 Centro Inteligente de Operações de Mobilidade Urbana")
st.caption("Classificação de condição operacional • Detecção de anomalias • Recomendação de ação")

# ---------- Sidebar: entrada manual ----------
st.sidebar.header("📥 Dados da via")

hora = st.sidebar.slider("Hora do dia", 0, 23, 8)
dia_semana = st.sidebar.selectbox(
    "Dia da semana",
    ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"],
)

st.sidebar.subheader("Fluxo de veículos")
volume = st.sidebar.slider("Volume (veículos/h)", 0, 6000, 2000)
velocidade = st.sidebar.slider("Velocidade média (km/h)", 0, 90, 40)
ocupacao = st.sidebar.slider("Ocupação da via (%)", 0, 100, 50)
tempo_viagem = st.sidebar.slider("Tempo médio de viagem (min)", 1, 120, 25)

st.sidebar.subheader("Transporte público")
atraso_onibus = st.sidebar.slider("Atraso dos ônibus (min)", 0, 60, 5)
intervalo_onibus = st.sidebar.slider("Intervalo entre ônibus (min)", 1, 40, 10)

st.sidebar.subheader("Condições externas")
chuva = st.sidebar.slider("Chuva (mm/h)", 0, 80, 0)
visibilidade = st.sidebar.slider("Visibilidade (m)", 50, 10000, 8000)

st.sidebar.subheader("Incidentes")
acidentes = st.sidebar.slider("Acidentes reportados", 0, 10, 0)
frenagens = st.sidebar.slider("Frenagens bruscas/h", 0, 300, 20)
falha_semaforica = st.sidebar.checkbox("Falha semafórica")
obra_na_via = st.sidebar.checkbox("Obra na via")
evento = st.sidebar.checkbox("Evento de grande porte")

# ---------- Montagem do registro ----------
registro = {
    "hora": hora,
    "dia_semana": dia_semana,
    "volume_veiculos_h": volume,
    "velocidade_media_kmh": velocidade,
    "ocupacao_via_pct": ocupacao,
    "tempo_medio_viagem_min": tempo_viagem,
    "atraso_onibus_min": atraso_onibus,
    "intervalo_onibus_min": intervalo_onibus,
    "chuva_mm_h": chuva,
    "visibilidade_m": visibilidade,
    "acidentes_reportados": acidentes,
    "falha_semaforica": int(falha_semaforica),
    "obra_na_via": int(obra_na_via),
    "evento_grande_porte": int(evento),
    "frenagens_bruscas_h": frenagens,
}

st.subheader("Registro atual")
st.dataframe(pd.DataFrame([registro]), use_container_width=True)

st.info("⏳ Etapa 2 adicionará o botão de predição.")