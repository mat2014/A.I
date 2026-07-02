"""
regras.py
Camada de inteligência computacional do Centro Inteligente de Operações
de Mobilidade Urbana — Sistema Especialista baseado em regras.

Papel na arquitetura:
- O classificador (ML) diz O QUE está acontecendo (classe + confiança).
- O detector de anomalias diz SE o padrão é atípico.
- Esta camada diz O QUE FAZER: aplica regras de engenharia de tráfego
  sobre as saídas dos modelos + variáveis brutas de contexto, produzindo
  uma recomendação operacional clara e JUSTIFICADA.

Por que um sistema especialista?
- Regras explícitas são auditáveis: o operador entende POR QUE o sistema
  recomendou cada ação (requisito de explicabilidade em operação crítica).
- Permite embutir conhecimento de engenharia que o modelo não aprende
  sozinho (ex.: chuva forte SEMPRE exige redução de velocidade,
  independentemente da classe prevista).
- Trata a incerteza do modelo: baixa confiança ou anomalia detectada
  levam a ações mais conservadoras (verificação humana).

Uso:
    from regras import decidir
    resultado = decidir(classe, confianca, anomalia, registro)

Executar demonstração:  python src/regras.py
"""

# Ações operacionais previstas no enunciado
ACOES = {
    "monitorar":      "Manter monitoramento",
    "semaforos":      "Ajustar tempos semafóricos",
    "campo":          "Acionar equipe de campo",
    "rota":           "Recomendar rota alternativa",
    "corredor":       "Priorizar corredor de ônibus",
    "alerta":         "Emitir alerta ao centro de operações",
    "reduzir_vel":    "Reduzir velocidade operacional da via",
}

# Limiares de engenharia (documentados para o PDF e a arguição)
LIMIAR_CHUVA_FORTE = 40      # mm/h  — acima disso, pista muito escorregadia
LIMIAR_VISIB_BAIXA = 1000    # m     — abaixo disso, visibilidade crítica
LIMIAR_ATRASO_ONIBUS = 15    # min   — acima disso, transporte público degradado
LIMIAR_CONFIANCA = 0.60      # confiança mínima para confiar na classe prevista
LIMIAR_FRENAGENS = 150       # frenagens/h — indício de comportamento de risco


def decidir(classe: str, confianca: float, anomalia: bool, registro: dict):
    """
    Aplica as regras do sistema especialista.

    Parâmetros
    ----------
    classe    : classe prevista pelo modelo de ML
    confianca : probabilidade da classe prevista (0 a 1)
    anomalia  : True se o Isolation Forest marcou o registro como anômalo
    registro  : dict com as variáveis brutas (chuva_mm_h, atraso_onibus_min,
                visibilidade_m, falha_semaforica, acidentes_reportados,
                frenagens_bruscas_h, evento_grande_porte, obra_na_via)

    Retorna
    -------
    dict com:  acoes (lista), justificativas (lista), nivel_risco (str)
    """
    acoes = []
    justif = []

    chuva = registro.get("chuva_mm_h", 0)
    visib = registro.get("visibilidade_m", 10000)
    atraso = registro.get("atraso_onibus_min", 0)
    falha = registro.get("falha_semaforica", 0)
    acidentes = registro.get("acidentes_reportados", 0)
    frenagens = registro.get("frenagens_bruscas_h", 0)
    evento = registro.get("evento_grande_porte", 0)
    obra = registro.get("obra_na_via", 0)

    # ------------------------------------------------------------------
    # BLOCO 1 — Regras por classe prevista
    # ------------------------------------------------------------------
    if classe == "FLUXO NORMAL":
        acoes.append(ACOES["monitorar"])
        justif.append("Classe prevista: fluxo normal, sem impacto relevante.")

    elif classe == "LENTIDÃO MODERADA":
        acoes.append(ACOES["semaforos"])
        justif.append("Lentidão moderada: ajuste semafórico pode "
                      "restabelecer a fluidez antes do agravamento.")

    elif classe == "CONGESTIONAMENTO CRÍTICO":
        acoes.append(ACOES["rota"])
        justif.append("Congestionamento crítico: desviar demanda é "
                      "prioritário para evitar colapso local.")
        acoes.append(ACOES["semaforos"])
        justif.append("Reprogramação semafórica ajuda a escoar a retenção.")

    elif classe == "INCIDENTE OPERACIONAL":
        acoes.append(ACOES["campo"])
        justif.append("Incidente operacional: verificação presencial "
                      "é necessária.")
        acoes.append(ACOES["alerta"])
        justif.append("Centro de operações deve ser notificado do incidente.")

    # ------------------------------------------------------------------
    # BLOCO 2 — Regras de contexto (independem da classe)
    # ------------------------------------------------------------------
    if chuva >= LIMIAR_CHUVA_FORTE or visib <= LIMIAR_VISIB_BAIXA:
        acoes.append(ACOES["reduzir_vel"])
        if chuva >= LIMIAR_CHUVA_FORTE:
            justif.append(f"Chuva intensa ({chuva:.0f} mm/h ≥ "
                          f"{LIMIAR_CHUVA_FORTE}): pista escorregadia.")
        if visib <= LIMIAR_VISIB_BAIXA:
            justif.append(f"Visibilidade baixa ({visib:.0f} m ≤ "
                          f"{LIMIAR_VISIB_BAIXA}): risco de colisão.")

    if atraso >= LIMIAR_ATRASO_ONIBUS:
        acoes.append(ACOES["corredor"])
        justif.append(f"Atraso de ônibus elevado ({atraso:.0f} min ≥ "
                      f"{LIMIAR_ATRASO_ONIBUS}): priorizar transporte "
                      f"coletivo reduz o impacto por passageiro.")

    if falha == 1:
        acoes.append(ACOES["campo"])
        justif.append("Falha semafórica reportada: requer manutenção em campo.")

    if acidentes >= 1 or frenagens >= LIMIAR_FRENAGENS:
        acoes.append(ACOES["alerta"])
        if acidentes >= 1:
            justif.append(f"{acidentes:.0f} acidente(s) reportado(s).")
        if frenagens >= LIMIAR_FRENAGENS:
            justif.append(f"Frenagens bruscas elevadas ({frenagens:.0f}/h): "
                          f"indício de comportamento de risco na via.")

    if evento == 1:
        acoes.append(ACOES["corredor"])
        justif.append("Evento de grande porte: reforçar transporte coletivo.")

    if obra == 1:
        acoes.append(ACOES["rota"])
        justif.append("Obra na via: capacidade reduzida, desviar demanda.")

    # ------------------------------------------------------------------
    # BLOCO 3 — Regras de incerteza (anomalia e baixa confiança)
    # ------------------------------------------------------------------
    if anomalia:
        acoes.append(ACOES["alerta"])
        justif.append("Padrão anômalo detectado (Isolation Forest): "
                      "situação atípica, requer atenção do operador.")
        if classe != "FLUXO NORMAL":
            acoes.append(ACOES["campo"])
            justif.append("Anomalia em condição adversa: verificação "
                          "presencial recomendada.")

    if confianca < LIMIAR_CONFIANCA:
        acoes.append(ACOES["alerta"])
        justif.append(f"Confiança do modelo baixa ({confianca:.0%} < "
                      f"{LIMIAR_CONFIANCA:.0%}): classificação incerta, "
                      f"validar com operador humano.")

    # ------------------------------------------------------------------
    # Nível de risco consolidado (para exibição na interface)
    # ------------------------------------------------------------------
    pontos = 0
    pontos += {"FLUXO NORMAL": 0, "LENTIDÃO MODERADA": 1,
               "CONGESTIONAMENTO CRÍTICO": 2,
               "INCIDENTE OPERACIONAL": 3}.get(classe, 1)
    pontos += 1 if anomalia else 0
    pontos += 1 if chuva >= LIMIAR_CHUVA_FORTE else 0
    pontos += 1 if (acidentes >= 1 or falha == 1) else 0

    if pontos == 0:
        nivel = "BAIXO"
    elif pontos <= 2:
        nivel = "MÉDIO"
    elif pontos <= 4:
        nivel = "ALTO"
    else:
        nivel = "CRÍTICO"

    # remove ações duplicadas preservando a ordem de prioridade
    acoes_unicas = list(dict.fromkeys(acoes))

    return {
        "acoes": acoes_unicas,
        "justificativas": justif,
        "nivel_risco": nivel,
    }


# ----------------------------------------------------------------------
# Demonstração com 4 cenários (executar: python src/regras.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    cenarios_teste = [
        ("Operação tranquila",
         "FLUXO NORMAL", 0.95, False,
         {"chuva_mm_h": 0, "atraso_onibus_min": 2}),

        ("Pico com chuva forte e ônibus atrasados",
         "CONGESTIONAMENTO CRÍTICO", 0.88, False,
         {"chuva_mm_h": 55, "visibilidade_m": 600,
          "atraso_onibus_min": 22}),

        ("Acidente grave e anômalo de madrugada",
         "INCIDENTE OPERACIONAL", 0.91, True,
         {"acidentes_reportados": 3, "frenagens_bruscas_h": 220,
          "chuva_mm_h": 20}),

        ("Situação ambígua (baixa confiança)",
         "LENTIDÃO MODERADA", 0.48, False,
         {"chuva_mm_h": 10, "atraso_onibus_min": 8}),
    ]

    for titulo, classe, conf, anom, reg in cenarios_teste:
        r = decidir(classe, conf, anom, reg)
        print("=" * 70)
        print(f"CENÁRIO: {titulo}")
        print(f"  Classe: {classe} | Confiança: {conf:.0%} | "
              f"Anomalia: {'SIM' if anom else 'NÃO'}")
        print(f"  Nível de risco: {r['nivel_risco']}")
        print("  Ações recomendadas:")
        for a in r["acoes"]:
            print(f"    - {a}")
        print("  Justificativas:")
        for j in r["justificativas"]:
            print(f"    * {j}")
    print("=" * 70)