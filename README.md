# Sistema Inteligente de Apoio à Decisão — Centro Inteligente de Operações de Mobilidade Urbana
 
**FIAP — Inteligência Artificial e Computacional**
**Autor:** Matheus Pinheiro — RM 83129
**São Paulo — 2026**
 
---
 
## Sumário
 
1. [Introdução](#1-introdução)
2. [Base de dados](#2-base-de-dados)
3. [Pré-processamento e análise exploratória](#3-pré-processamento-e-análise-exploratória)
4. [Modelos de classificação](#4-modelos-de-classificação)
5. [Detecção de anomalias](#5-detecção-de-anomalias)
6. [Camada de inteligência computacional](#6-camada-de-inteligência-computacional)
7. [Arquitetura do sistema](#7-arquitetura-do-sistema)
8. [Interface de demonstração](#8-interface-de-demonstração)
9. [Análise crítica dos resultados](#9-análise-crítica-dos-resultados)
10. [Limitações da solução](#10-limitações-da-solução)
11. [Conclusão](#11-conclusão)
12. [Referências](#referências)
---
 
## 1. Introdução
 
Grandes centros urbanos como São Paulo dependem de decisões operacionais tomadas em janelas de poucos minutos. Uma falha semafórica em corredor estruturante, uma chuva intensa em horário de pico ou um acidente em via arterial produzem efeitos que se propagam rapidamente pela malha, afetando tempo de deslocamento, regularidade do transporte público e resposta de serviços essenciais. O custo de uma decisão tardia é alto; o de uma decisão errada, também.
 
Este trabalho apresenta um sistema inteligente de apoio à decisão desenvolvido para o Centro Inteligente de Operações de Mobilidade Urbana. O sistema recebe as variáveis operacionais de um corredor monitorado, classifica sua condição em um dos quatro estados operacionais definidos no escopo, verifica se o comportamento observado é atípico frente ao padrão histórico e traduz esse diagnóstico em uma recomendação de ação, acompanhada da justificativa que a sustenta.
 
O ponto central da solução não é o classificador isoladamente. Um modelo supervisionado informa qual é o estado da via, mas não informa se aquela combinação de variáveis é conhecida, nem o que fazer a respeito. A arquitetura foi por isso construída em três camadas complementares: um **classificador supervisionado**, um **detector de anomalias não supervisionado** e um **sistema especialista baseado em regras de engenharia**. Cada camada responde a uma pergunta distinta, e as três juntas produzem uma saída acionável e auditável — requisito essencial em contexto operacional crítico, no qual o operador humano precisa entender por que o sistema recomendou determinada ação antes de executá-la.
 
### 1.1. Objetivos
 
- Construir uma base de dados operacional coerente, documentada e reprodutível.
- Implementar um pipeline completo de pré-processamento e análise exploratória.
- Treinar e comparar três modelos de classificação, justificando a escolha final por métricas.
- Implementar uma camada de detecção de anomalias que complemente a classificação.
- Implementar uma camada de inteligência computacional que converta o diagnóstico em ação.
- Disponibilizar uma interface funcional para teste e demonstração.
- Analisar criticamente os resultados e explicitar as limitações da solução.
### 1.2. Visão geral
 
| Etapa | Componente | Saída produzida |
|---|---|---|
| 1 | `dados/base_original.csv` | 1.500 registros, 15 variáveis de entrada |
| 2 | Pré-processamento | Base tratada, imputada, codificada e padronizada |
| 3 | Classificação — Random Forest | Estado operacional + probabilidade por classe |
| 4 | Detecção de anomalias — Isolation Forest | Indicação binária + score de anomalia |
| 5 | Sistema especialista | Ações recomendadas + nível de risco + justificativa |
 
A separação entre as etapas 3, 4 e 5 é deliberada. Classificação e detecção de anomalias operam de forma independente e podem discordar entre si — e é justamente essa discordância que carrega informação operacional relevante, como discutido na seção 9.
 
---
 
## 2. Base de dados
 
A base é sintética, gerada pelo script `src/gerar_dados.py` no arquivo `dados/base_original.csv`, e contém **1.500 registros** — acima do mínimo de 1.000 exigido. A opção pela geração sintética foi tomada por três razões técnicas: garante controle sobre os cenários representados, assegura a presença de todos os casos críticos exigidos e torna a base integralmente reprodutível a partir do código-fonte, com semente fixa (`random_state=42`).
 
### 2.1. Variáveis monitoradas
 
Cada registro representa uma leitura consolidada de um corredor monitorado em janela de uma hora, agregando quatro grupos de variáveis: fluxo de veículos, transporte público, condições externas e incidentes/segurança.
 
| Variável | Unidade | Faixa | Grupo |
|---|---|---|---|
| `hora` | hora | 0 – 23 | Temporal |
| `dia_semana` | categoria | segunda – domingo | Temporal |
| `volume_veiculos_h` | veículos/hora | 0 – 6000 | Fluxo |
| `velocidade_media_kmh` | km/h | 0 – 90 | Fluxo |
| `ocupacao_via_pct` | % | 0 – 100 | Fluxo |
| `tempo_medio_viagem_min` | minutos | 1 – 120 | Fluxo |
| `atraso_onibus_min` | minutos | 0 – 60 | Transporte público |
| `intervalo_onibus_min` | minutos | 1 – 40 | Transporte público |
| `chuva_mm_h` | mm/h | 0 – 80 | Condições externas |
| `visibilidade_m` | metros | 50 – 10000 | Condições externas |
| `obra_na_via` | binário | 0 ou 1 | Condições externas |
| `evento_grande_porte` | binário | 0 ou 1 | Condições externas |
| `acidentes_reportados` | quantidade | 0 – 10 | Incidentes |
| `falha_semaforica` | binário | 0 ou 1 | Incidentes |
| `frenagens_bruscas_h` | quantidade/hora | 0 – 300 | Incidentes |
| `status_operacional` | classe | 4 categorias | Variável alvo |
 
A base entregue inclui ainda a coluna auxiliar `cenario_origem`, que registra qual dos seis cenários gerou cada linha. Essa coluna **não é usada no treinamento** — sua presença serviria como vazamento direto do rótulo — mas é fundamental para auditar a base e para avaliar o comportamento do detector de anomalias, como feito na seção 5. A descrição completa de todas as variáveis está no arquivo `DICIONARIO_DADOS.md`.
 
### 2.2. Estados operacionais
 
| Classe | Descrição operacional |
|---|---|
| **FLUXO NORMAL** | Trânsito estável. Velocidade próxima da velocidade de projeto, ocupação baixa e ausência de ocorrências. |
| **LENTIDÃO MODERADA** | Redução de velocidade e aumento do tempo de deslocamento, com ocupação elevada mas sem colapso. Típico de pico ou chuva moderada. |
| **CONGESTIONAMENTO CRÍTICO** | Alto impacto operacional. Ocupação próxima da saturação, velocidade muito baixa e risco de colapso local do corredor. |
| **INCIDENTE OPERACIONAL** | Condição anormal causada por acidente, falha semafórica, obra ou evento. Caracteriza-se pelo comportamento atípico das variáveis, não apenas por volume elevado. |
 
A distinção entre CONGESTIONAMENTO CRÍTICO e INCIDENTE OPERACIONAL é conceitualmente a mais delicada da base. Ambos apresentam velocidade baixa e ocupação alta; o que os separa é a causa. O congestionamento decorre de excesso de demanda sobre a capacidade da via — situação previsível e recorrente —, enquanto o incidente decorre de evento externo que reduz abruptamente a capacidade disponível. Operacionalmente a diferença é decisiva: o primeiro pede gestão de fluxo e rotas alternativas; o segundo pede acionamento de equipe de campo.
 
### 2.3. Lógica de geração dos dados
 
A base não foi gerada por sorteio uniforme. Cada registro é produzido a partir de um dos seis cenários causais abaixo, nos quais as variáveis são geradas de forma condicionada umas às outras, respeitando relações físicas e operacionais conhecidas do tráfego urbano.
 
| Cenário | Registros | Lógica aplicada às variáveis |
|---|---|---|
| normal | 600 | Volume baixo a médio, velocidade alta, ocupação baixa, sem chuva relevante nem ocorrências. Transporte público dentro da tabela programada. |
| pico | 300 | Volume elevado nas faixas de 7h–9h e 17h–19h em dias úteis. Ocupação alta, velocidade reduzida por saturação e atraso crescente dos ônibus. |
| chuva | 180 | Precipitação elevada com queda proporcional da visibilidade, redução de velocidade e aumento de frenagens bruscas. |
| falha | 150 | `falha_semaforica` ativada, com retenção acentuada: ocupação e tempo de viagem sobem sem que o volume de entrada seja o mais alto da base. |
| acidente | 150 | `acidentes_reportados` maior que zero, pico acentuado de frenagens bruscas e queda abrupta de velocidade, desacoplada do volume. |
| evento | 120 | `evento_grande_porte` ativado, com volume elevado fora do padrão horário habitual e forte impacto no transporte público. |
 
As relações entre variáveis seguem a lógica do domínio. A velocidade média é derivada da ocupação segundo relação decrescente, refletindo o comportamento do diagrama fundamental do tráfego. O tempo médio de viagem é calculado a partir da velocidade resultante, mantendo coerência interna entre as duas grandezas. A visibilidade é reduzida em função da precipitação. O atraso dos ônibus responde tanto à ocupação da via quanto à presença de eventos e incidentes.
 
Os seis cenários não correspondem um a um às quatro classes. O cenário de pico pode produzir tanto LENTIDÃO MODERADA quanto CONGESTIONAMENTO CRÍTICO, dependendo da severidade sorteada; o cenário de chuva pode produzir FLUXO NORMAL sob precipitação leve. Essa relação de muitos-para-muitos é o que impede a base de ser trivialmente separável.
 
### 2.4. Ruído e imperfeições intencionais
 
Uma base sintética construída apenas com regras determinísticas produziria classes perfeitamente separáveis e métricas próximas de 100% — resultado que indicaria problema de construção, e não qualidade do modelo. Duas perturbações foram aplicadas deliberadamente:
 
| Perturbação | Aplicação | Justificativa |
|---|---|---|
| Ruído de rótulo | Parte dos registros teve o `status_operacional` trocado por classe adjacente. | Simula a divergência de julgamento entre operadores humanos em situações de fronteira, sobretudo entre lentidão e congestionamento. |
| Valores ausentes | 144 valores ausentes distribuídos em cinco variáveis contínuas de sensores: `ocupacao_via_pct` (40), `visibilidade_m` (31), `atraso_onibus_min` (28), `velocidade_media_kmh` (26) e `chuva_mm_h` (19). | Simula falha de sensor e perda de telemetria, rotineiras em operação real, e força o pipeline a tratar dados incompletos. |
 
Essas imperfeições são a razão técnica pela qual as métricas se situam em patamar realista, e não em valores artificialmente perfeitos.
 
---
 
## 3. Pré-processamento e análise exploratória
 
O pré-processamento e a análise exploratória estão implementados no notebook `notebooks/analise.ipynb`, que lê a base original em `.csv` e produz a base tratada utilizada no treinamento.
 
### 3.1. Etapas do pipeline
 
| Etapa | Tratamento aplicado | Decisão técnica |
|---|---|---|
| Leitura | Carga de `dados/base_original.csv` com pandas. | Verificação de tipos, faixas e consistência das variáveis. |
| Dados ausentes | Imputação pela mediana nas cinco variáveis contínuas afetadas. | Descartar os 144 registros incompletos reduziria a base e enviesaria a amostra, já que a ausência não é uniforme entre cenários. A mediana é robusta a valores extremos, frequentes nos cenários críticos. |
| Variáveis categóricas | `dia_semana` convertida por `get_dummies` com `dtype=int`. | Não há ordem natural entre os dias, o que desaconselha codificação ordinal. |
| Padronização | `StandardScaler` nas variáveis contínuas. | As escalas são muito distintas (visibilidade em milhares de metros, chuva em dezenas). A MLP e modelos sensíveis a distância exigem escalas comparáveis. |
| Separação | Divisão treino / validação / teste, estratificada pelo alvo. | A validação orienta a comparação entre modelos; o teste, mantido intocado, fornece a estimativa final de desempenho. |
| Persistência | Ordem das colunas salva em `modelos/colunas.pkl`. | Garante que a entrada montada pela interface reproduza exatamente o formato usado no treino. |
 
O último item merece ênfase por ser fonte frequente de falha silenciosa em sistemas de ML em produção: se a ordem ou o conjunto de colunas gerado na inferência divergir do usado no treino, o modelo retorna previsões sem erro aparente, porém incorretas. Persistir explicitamente o vetor de colunas elimina esse risco.
 
> **Ressalva metodológica sobre a imputação.** A mediana usada para preencher os valores ausentes é calculada sobre a base completa, antes da separação entre treino, validação e teste. Isso constitui um vazamento leve: a mediana de cada variável incorpora informação dos registros que depois compõem o conjunto de teste. O procedimento tecnicamente correto seria encapsular a imputação em um `Pipeline` do scikit-learn, ajustado exclusivamente sobre o treino, como já é feito com o `StandardScaler`. O impacto prático é reduzido — são 144 valores em 1.500 registros, ou 1,9% das células das cinco colunas afetadas, e a mediana é uma estatística robusta —, mas a limitação é real e está registrada na seção 10.
 
### 3.2. Distribuição das classes
 
![Figura 1 — Distribuição dos registros por estado operacional](docs/figuras/figura_01_distribuicao_classes.png)
 
| Classe | Registros | % |
|---|---|---|
| FLUXO NORMAL | 565 | 37,7% |
| LENTIDÃO MODERADA | 347 | 23,1% |
| CONGESTIONAMENTO CRÍTICO | 207 | 13,8% |
| INCIDENTE OPERACIONAL | 381 | 25,4% |
 
A base é desbalanceada por construção, refletindo a frequência relativa esperada em operação real. Esse desbalanceamento é a razão técnica pela qual o **F1-score macro** foi adotado como métrica principal de comparação, e não a acurácia: um modelo que ignorasse completamente a classe minoritária ainda alcançaria acurácia próxima de 86%.
 
### 3.3. Correlação entre variáveis
 
![Figura 2 — Matriz de correlação entre as variáveis numéricas](docs/figuras/figura_02_correlacao.png)
 
A matriz confirma as relações esperadas do domínio: correlação negativa forte entre velocidade média e ocupação da via, correlação negativa entre velocidade e tempo médio de viagem, e correlação positiva entre chuva e frenagens bruscas, acompanhada de correlação negativa entre chuva e visibilidade. Igualmente relevante é o que não aparece: nenhuma variável preditora exibe correlação próxima de 1,0 com outra, o que indica ausência de redundância perfeita e, sobretudo, ausência de vazamento de informação do rótulo para as preditoras.
 
### 3.4. Comportamento das variáveis por classe
 
![Figura 3 — Distribuição das principais variáveis por estado operacional](docs/figuras/figura_03_boxplots.png)
 
Os boxplots evidenciam dois padrões que antecipam o comportamento do classificador. Primeiro, a sobreposição parcial entre LENTIDÃO MODERADA e CONGESTIONAMENTO CRÍTICO nas variáveis de fluxo — velocidade, ocupação e tempo de viagem —, indicando que essas duas classes ocupam regiões contíguas do espaço de variáveis. Segundo, a separação nítida do INCIDENTE OPERACIONAL na variável de frenagens bruscas, que se comporta como assinatura característica dessa classe. A previsão decorrente é que os erros do modelo se concentrem na primeira fronteira e sejam raros na segunda, o que se confirma na matriz de confusão da seção 4.
 
---
 
## 4. Modelos de classificação
 
### 4.1. Modelos testados
 
Foram treinados e comparados três modelos, escolhidos por representarem famílias algorítmicas distintas — modelo baseado em regras, ensemble e rede neural — de modo que a comparação informe sobre a natureza do problema, e não apenas sobre a variação entre implementações semelhantes.
 
| Modelo | Família | Motivação da escolha |
|---|---|---|
| Árvore de Decisão | Modelo baseado em regras | Linha de base interpretável. Suas divisões podem ser lidas diretamente como limiares operacionais, o que permite verificar se o modelo aprendeu critérios plausíveis. |
| Random Forest | Ensemble de árvores (bagging) | Reduz a variância da árvore individual, trata bem relações não lineares e fornece importância de variáveis e probabilidades por classe. |
| MLP (rede neural) | Rede neural multicamada | Verifica se há ganho em modelar fronteiras de decisão contínuas e altamente não lineares, inacessíveis a modelos baseados em divisões ortogonais. |
 
### 4.2. Comparação no conjunto de validação
 
A métrica principal adotada foi o **F1-score macro**. A escolha se justifica porque a acurácia isolada é enganosa em problema multiclasse desbalanceado com classes de importância operacional desigual: um modelo que acerte bem as classes frequentes e erre sistematicamente o CONGESTIONAMENTO CRÍTICO — a classe mais rara — exibiria acurácia alta e ainda assim seria inadequado à operação. O F1-macro atribui peso igual a todas as classes, penalizando esse comportamento.
 
| Modelo | Acurácia | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Árvore de Decisão | 0,881 | 0,865 | 0,860 | 0,862 |
| **Random Forest** | **0,929** | **0,926** | **0,903** | **0,913** |
| MLP (rede neural) | 0,836 | 0,817 | 0,811 | 0,813 |
 
*Tabela 1 — Desempenho dos três modelos no conjunto de validação.*
 
### 4.3. Análise comparativa
 
O Random Forest supera a Árvore de Decisão em cerca de 5 pontos percentuais de F1-macro. O ganho é coerente com a construção da base: o ruído de rótulo introduzido faz com que uma árvore isolada memorize registros inconsistentes durante o treino, ao passo que a agregação de múltiplas árvores treinadas em amostras distintas dilui esse efeito. O ensemble, nesse sentido, não apenas melhora a métrica — ele responde diretamente à imperfeição deliberadamente inserida nos dados.
 
O resultado mais informativo da comparação, contudo, é o desempenho inferior da MLP, com F1-macro de 0,813, abaixo inclusive da árvore simples. Isso não indica falha da rede neural, e sim que a estrutura do problema favorece modelos baseados em divisões. As classes operacionais são definidas, na origem, por limiares sobre variáveis — ocupação acima de determinado valor, presença de acidentes, precipitação acima de um patamar. Modelos de árvore representam esse tipo de fronteira de forma nativa, com uma única divisão. Uma rede neural precisa aproximar cada limiar por combinação de funções contínuas, o que exige mais dados do que os 1.500 registros disponíveis. Com uma base maior, a diferença tenderia a diminuir; com esta base, a escolha de arquitetura mais simples é a tecnicamente correta.
 
### 4.4. Modelo selecionado e desempenho no teste
 
O Random Forest foi selecionado como modelo final e reavaliado no conjunto de teste, mantido intocado durante toda a seleção. O desempenho obtido foi de **0,933 de acurácia** e **0,919 de F1-macro** sobre 225 registros — ligeiramente superior ao observado na validação, o que indica ausência de sobreajuste ao conjunto usado para a escolha.
 
| Classe | Precision | Recall | F1-score | Suporte |
|---|---|---|---|---|
| FLUXO NORMAL | 0,954 | 0,976 | 0,965 | 85 |
| LENTIDÃO MODERADA | 0,902 | 0,885 | 0,893 | 52 |
| CONGESTIONAMENTO CRÍTICO | 0,926 | 0,806 | 0,862 | 31 |
| INCIDENTE OPERACIONAL | 0,933 | 0,982 | 0,957 | 57 |
| **Média macro** | 0,929 | 0,912 | 0,919 | 225 |
| **Acurácia global** | — | — | 0,933 | 225 |
 
*Tabela 2 — Desempenho do Random Forest por classe, no conjunto de teste.*
 
A leitura por classe é mais informativa que a média. O recall de 0,982 no INCIDENTE OPERACIONAL significa que o modelo deixou de detectar apenas 1 dos 57 incidentes reais do conjunto de teste — desempenho adequado, já que esta é a classe cuja omissão tem maior custo operacional. Em contrapartida, o recall de 0,806 no CONGESTIONAMENTO CRÍTICO é o ponto fraco do modelo: cerca de um em cada cinco congestionamentos críticos não é reconhecido como tal. Trata-se, simultaneamente, da classe com menos registros na base e da que ocupa a região de fronteira mais ambígua.
 
### 4.5. Matriz de confusão
 
![Figura 4 — Matriz de confusão do Random Forest no conjunto de teste](docs/figuras/figura_04_matriz_confusao.png)
 
### 4.6. Análise dos principais erros
 
O modelo cometeu 15 erros em 225 previsões. A distribuição desses erros é mais relevante que o total:
 
| Erro | Ocorrências | Gravidade operacional |
|---|---|---|
| LENTIDÃO MODERADA prevista como FLUXO NORMAL | 4 | Moderada. O sistema subestima a condição e deixa de recomendar ajuste semafórico. |
| CONGESTIONAMENTO CRÍTICO previsto como LENTIDÃO MODERADA | 3 | Moderada. Subestima a severidade, mas as ações recomendadas são parcialmente sobrepostas. |
| CONGESTIONAMENTO CRÍTICO previsto como INCIDENTE OPERACIONAL | 3 | Baixa. Superestima a severidade, acionando equipe de campo sem necessidade. |
| FLUXO NORMAL previsto como LENTIDÃO MODERADA | 2 | Baixa. Falso alarme conservador. |
| LENTIDÃO MODERADA prevista como CONGESTIONAMENTO CRÍTICO ou INCIDENTE | 2 | Baixa. Superestima a severidade. |
| INCIDENTE OPERACIONAL previsto como CONGESTIONAMENTO CRÍTICO | 1 | **Alta.** É o erro mais grave do sistema: deixa de acionar a equipe de campo. |
 
Onze dos quinze erros ocorrem entre classes adjacentes na escala de severidade, confirmando a previsão feita na análise exploratória: as confusões se concentram na fronteira contínua entre lentidão e congestionamento. Não existe um valor exato de ocupação a partir do qual a lentidão vira congestionamento, e o ruído de rótulo aplicado na geração reproduz precisamente essa ambiguidade. Operacionalmente, esse é o erro menos danoso, pois as ações recomendadas para os dois estados são adjacentes e parcialmente sobrepostas.
 
O erro crítico é o registro isolado em que um INCIDENTE OPERACIONAL real foi classificado como CONGESTIONAMENTO CRÍTICO, pois deixaria de acionar a equipe de campo. É precisamente essa falha que a camada de detecção de anomalias existe para mitigar, funcionando como verificação independente do classificador.
 
### 4.7. Importância das variáveis
 
![Figura 5 — Importância das variáveis no Random Forest](docs/figuras/figura_05_importancia_variaveis.png)
 
O ranking de importância confirma que o modelo aprendeu critérios plausíveis do ponto de vista da engenharia de tráfego, e não correlações espúrias. As três primeiras posições são ocupadas pelas variáveis de estado do fluxo — velocidade média (0,188), ocupação da via (0,147) e tempo médio de viagem (0,126) —, que somam 46% da importância total e correspondem exatamente às grandezas que um operador humano observaria primeiro.
 
A quarta posição das frenagens bruscas (0,112) é o achado mais interessante: trata-se de uma variável de segurança, não de fluxo, e sua relevância elevada mostra que o modelo a utiliza como marcador discriminante do INCIDENTE OPERACIONAL — coerente com a separação nítida observada no boxplot da Figura 3. Por outro lado, a baixa importância dos acidentes reportados (0,026) tem explicação direta: por serem raros e binários na prática, carregam pouca informação em termos de ganho médio de impureza, embora sejam altamente decisivos nos poucos casos em que ocorrem. Essa é uma limitação conhecida da métrica de importância por impureza, e a razão pela qual acidentes reportados aparecem explicitamente como regra no sistema especialista da seção 6, independentemente de seu peso no classificador.
 
---
 
## 5. Detecção de anomalias
 
### 5.1. Técnica adotada
 
A camada de detecção de anomalias foi implementada com **Isolation Forest**, configurado com contaminação de 5% e semente fixa. O algoritmo isola observações construindo árvores com divisões aleatórias: pontos anômalos, por estarem em regiões esparsas do espaço de variáveis, exigem menos divisões para serem isolados e recebem pontuação de anomalia mais alta.
 
A escolha se justifica por três características adequadas ao problema. O algoritmo não exige rótulos, permitindo identificar situações atípicas que não correspondem a nenhuma classe conhecida. Tem complexidade baixa e opera bem em dados multidimensionais, condição necessária para uso em tempo real. E, ao contrário de métodos baseados em densidade como o DBSCAN, não exige definição prévia de parâmetros de vizinhança sensíveis à escala.
 
### 5.2. Resultados obtidos
 
O detector sinalizou 75 dos 1.500 registros como anômalos — os 5% esperados pela configuração de contaminação.
 
| Classe operacional | Registros | Anômalos | % da classe |
|---|---|---|---|
| FLUXO NORMAL | 565 | 1 | 0,2% |
| LENTIDÃO MODERADA | 347 | 5 | 1,4% |
| CONGESTIONAMENTO CRÍTICO | 207 | 3 | 1,4% |
| INCIDENTE OPERACIONAL | 381 | 66 | 17,3% |
 
*Tabela 3 — Anomalias detectadas por estado operacional.*
 
Dos 75 registros sinalizados, **66 pertencem à classe INCIDENTE OPERACIONAL — 88% do total**. O resultado é notável porque o detector foi treinado sem qualquer acesso ao rótulo: ele identificou, apenas a partir da geometria dos dados, a mesma região do espaço de variáveis que corresponde à classe operacionalmente mais crítica. Isso valida tanto a coerência interna da base quanto a adequação da técnica ao problema.
 
| Cenário de origem | Registros | Anômalos | % do cenário |
|---|---|---|---|
| acidente | 150 | 37 | 24,7% |
| evento | 120 | 33 | 27,5% |
| chuva | 180 | 3 | 1,7% |
| falha | 150 | 2 | 1,3% |
| pico | 300 | 0 | 0,0% |
| normal | 600 | 0 | 0,0% |
 
*Tabela 4 — Anomalias detectadas por cenário de geração.*
 
A leitura por cenário revela o comportamento mais importante do detector: nenhum dos 900 registros dos cenários normal e pico foi sinalizado como anômalo. Em termos operacionais, isso significa taxa nula de falso alarme na operação de rotina — inclusive no horário de pico, situação em que as variáveis assumem valores extremos (volume alto, velocidade baixa, ocupação elevada) mas de forma perfeitamente previsível. O detector distingue corretamente entre valor extremo e combinação atípica, que é exatamente a distinção que se espera dele.
 
Em contrapartida, os cenários de acidente e evento concentram as detecções, com aproximadamente um quarto de seus registros sinalizados. Chuva e falha semafórica são pouco detectados, o que também é coerente: ambos produzem degradação previsível e proporcional das variáveis, sem romper o padrão de relações entre elas.
 
### 5.3. Distribuição dos scores
 
![Figura 6 — Distribuição do score de anomalia por estado operacional](docs/figuras/figura_06_scores_anomalia.png)
 
O gráfico mostra a mediana do score decrescendo consistentemente do FLUXO NORMAL ao INCIDENTE OPERACIONAL, com o incidente sendo a única classe cuja caixa inteira se aproxima do limiar de decisão. A LENTIDÃO MODERADA apresenta a maior dispersão entre as quatro classes, com a caixa se estendendo de aproximadamente 0,036 a 0,101 — reflexo direto de ser a classe gerada pela maior variedade de cenários de origem, incluindo pico, chuva e casos de fronteira.
 
### 5.4. Complementaridade com a classificação
 
As duas camadas respondem a perguntas diferentes, e é essa diferença que gera valor. O classificador responde: *em qual dos quatro estados conhecidos esta situação melhor se enquadra?* Por construção, ele sempre devolve uma das quatro classes, ainda que a situação observada não se pareça com nada visto no treinamento. O detector responde: *esta combinação de variáveis é compatível com o padrão histórico?* — pergunta que o classificador não é capaz de formular.
 
| Classificação | Anomalia | Interpretação operacional |
|---|---|---|
| FLUXO NORMAL | Não | Situação normal e reconhecida. Manter monitoramento de rotina. |
| FLUXO NORMAL | Sim | **Alerta preventivo.** Indicadores agregados normais, mas combinação atípica — possível degradação inicial ou falha de sensor. |
| CONGESTIONAMENTO CRÍTICO | Não | Congestionamento recorrente e previsível. Aplicar gestão de fluxo e rota alternativa. |
| CONGESTIONAMENTO CRÍTICO | Sim | **Congestionamento fora do padrão histórico.** Eleva o nível de risco: a causa não se explica apenas por demanda. |
| INCIDENTE OPERACIONAL | Sim | Confirmação cruzada por duas camadas independentes. Máxima prioridade de resposta. |
 
A segunda e a quarta linhas representam o ganho concreto da arquitetura em duas camadas: são situações que um classificador isolado trataria como rotina ou como congestionamento comum, sem qualquer sinalização ao operador.
 
---
 
## 6. Camada de inteligência computacional
 
### 6.1. Técnica adotada
 
A camada de decisão foi implementada como **sistema especialista baseado em regras**, no módulo `src/regras.py`. A função `decidir(classe, confianca, anomalia, registro)` recebe a saída das duas camadas anteriores acrescida das variáveis brutas do registro e retorna um dicionário com três campos: as ações recomendadas, o nível de risco atribuído e a lista de justificativas correspondentes às regras acionadas.
 
A opção pelo sistema especialista, em detrimento de lógica fuzzy ou algoritmo genético, decorre de um requisito do domínio: **rastreabilidade**. Em operação crítica, o operador precisa saber exatamente qual condição motivou cada recomendação, e um auditor precisa poder verificar a decisão posteriormente. Um conjunto de regras explícitas oferece essa propriedade de forma direta, e cada regra corresponde a um critério de engenharia de tráfego que pode ser discutido, validado e ajustado por especialista humano sem retreinamento de modelo.
 
Há ainda uma justificativa empírica, derivada da seção 4.7: a variável `acidentes_reportados` tem baixa importância no Random Forest (0,026) por ser rara, embora seja decisiva nos casos em que ocorre. O sistema especialista corrige essa assimetria ao tratá-la como regra explícita, independentemente do peso que o classificador lhe atribui. A camada de regras não apenas traduz o diagnóstico em ação — ela compensa uma limitação conhecida do modelo estatístico.
 
### 6.2. Estrutura da decisão
 
O sistema opera em três estágios sequenciais. O primeiro define a ação base a partir da classe prevista. O segundo aplica regras condicionais sobre as variáveis brutas, acrescentando ações específicas. O terceiro consolida o nível de risco, elevando-o quando há concordância entre múltiplos indicadores de severidade ou quando a confiança do classificador é baixa.
 
| Condição observada | Ação recomendada |
|---|---|
| Classe FLUXO NORMAL | Manter monitoramento |
| Classe LENTIDÃO MODERADA | Ajustar tempos semafóricos |
| Classe CONGESTIONAMENTO CRÍTICO | Recomendar rota alternativa e ajustar tempos semafóricos |
| Classe INCIDENTE OPERACIONAL | Emitir alerta ao centro de operações e acionar equipe de campo |
| Atraso elevado dos ônibus | Priorizar corredor de ônibus |
| Chuva intensa ou visibilidade reduzida | Reduzir velocidade operacional da via |
| Acidentes reportados ou falha semafórica | Acionar equipe de campo |
| Anomalia detectada | Elevar o nível de risco e registrar a ocorrência para revisão |
| Confiança do classificador abaixo do limiar | Sinalizar decisão de baixa confiança e recomendar validação humana |
 
As ações são **acumulativas, e não excludentes**. Uma mesma situação pode acionar várias regras simultaneamente, produzindo um conjunto de recomendações coerente com a complexidade real do evento.
 
### 6.3. Exemplo de decisão composta
 
Considere um registro classificado como CONGESTIONAMENTO CRÍTICO, com anomalia detectada, chuva intensa e atraso elevado dos ônibus. O sistema aciona quatro regras e consolida a seguinte saída:
 
- **Ações:** recomendar rota alternativa, ajustar tempos semafóricos, reduzir velocidade operacional da via e priorizar corredor de ônibus.
- **Nível de risco:** elevado, pela anomalia detectada somada à severidade da classe prevista.
- **Justificativas:** classe prevista com alta confiança; comportamento atípico frente ao padrão histórico; precipitação acima do limiar de segurança; atraso do transporte público acima do limite operacional.
A saída não é apenas a ação, mas o encadeamento que a produziu — permitindo ao operador concordar, discordar ou complementar a recomendação com informação de campo.
 
### 6.4. Justificativa frente às alternativas
 
| Técnica | Avaliação para este problema |
|---|---|
| **Sistema especialista (adotado)** | Rastreável, auditável e ajustável por especialista de domínio sem retreinamento. Integra-se diretamente às saídas das duas camadas anteriores e compensa a subponderação de variáveis raras pelo classificador. |
| Lógica fuzzy | Adequada à transição contínua entre lentidão e congestionamento — justamente onde se concentram 11 dos 15 erros do modelo. Exigiria, porém, calibração de funções de pertinência sem base empírica disponível neste projeto. |
| Algoritmo genético | Voltado à otimização de parâmetros, não à tradução de diagnóstico em ação. Não responde ao requisito desta camada. |
| Busca informada | Aplicável ao roteamento alternativo, que exigiria representação da malha viária — fora do escopo definido. |
 
---
 
## 7. Arquitetura do sistema
 
![Figura 7 — Fluxograma da arquitetura da solução](docs/figuras/figura_07_arquitetura.png)
 
```
ENTRADA DE DADOS (formulário na interface ou arquivo .csv)
        │
        ▼
PRÉ-PROCESSAMENTO (imputação · dummies · StandardScaler · reordenação por colunas.pkl)
        │
        ├───────────────────────────┐
        ▼                           ▼
CAMADA 1 — CLASSIFICAÇÃO     CAMADA 2 — ANOMALIA
Random Forest                Isolation Forest (contaminação 5%)
→ estado operacional         → comportamento atípico (sim/não)
→ probabilidade por classe   → score de anomalia
        │                           │
        └────────────┬──────────────┘
                     ▼   (+ variáveis brutas do registro)
CAMADA 3 — INTELIGÊNCIA COMPUTACIONAL
Sistema especialista baseado em regras
decidir(classe, confiança, anomalia, registro)
                     │
                     ▼
SAÍDA OPERACIONAL
Estado previsto · Nível de confiança · Indicação de anomalia
Ações recomendadas · Nível de risco · Justificativa da decisão
```
 
O fluxo parte da entrada de dados, por preenchimento manual na interface ou carga de arquivo, passa pelo pré-processamento que reproduz exatamente as transformações do treino, alimenta em paralelo o classificador e o detector de anomalias, e converge para o sistema especialista, que consolida a saída final apresentada ao operador.
 
---
 
## 8. Interface de demonstração
 
A interface foi implementada em **Streamlit**, no arquivo `src/app.py`, e opera em dois modos complementares. No **modo manual**, o operador ajusta as quinze variáveis operacionais nos controles do painel lateral e obtém a análise de um único registro. No **modo em lote**, carrega um arquivo `.csv` com vários registros e recebe a análise de todos de uma vez, com opção de exportar o resultado. Em ambos os modos a tela apresenta o estado operacional previsto, a probabilidade atribuída a cada uma das quatro classes, a indicação de anomalia acompanhada do score, as ações recomendadas e a justificativa completa da decisão.
 
O modo em lote reproduz exatamente o mesmo pré-processamento do modo manual, com uma diferença: os valores ausentes eventualmente presentes no arquivo são imputados pela mediana do próprio lote, e colunas extras — como `status_operacional` ou `cenario_origem` — são ignoradas. Isso permite submeter ao sistema registros operacionais em formato bruto, incluindo amostras da própria base, sem qualquer preparação prévia.
 
![Figura 8 — Tela de entrada](docs/figuras/figura_08_tela_entrada.png)
 
*Figura 8 — Tela de entrada: painel lateral com as quinze variáveis operacionais e registro montado a partir dos valores selecionados.*
 
### 8.1. Cenário 1 — Fluxo normal
 
Via operando dentro do padrão: velocidade de 60 km/h, ocupação de 35% e ausência de ocorrências. O classificador atribui FLUXO NORMAL com 98,9% de confiança, o detector não sinaliza anomalia (score 0,119) e o sistema especialista recomenda uma única ação — manter monitoramento —, com risco **BAIXO**.
 
![Figura 9 — Cenário de fluxo normal](docs/figuras/figura_09_cenario_fluxo_normal.png)
 
### 8.2. Cenário 2 — Incidente operacional com anomalia detectada
 
Acidente às 3h da madrugada, com três acidentes reportados, 220 frenagens bruscas por hora, chuva de 55 mm/h e visibilidade de 600 metros. Este é o cenário que melhor ilustra a arquitetura em três camadas.
 
O classificador prevê INCIDENTE OPERACIONAL, mas com confiança de apenas 61,5% — a combinação de volume baixo (típico de madrugada) com ocupação de 93% e velocidade de 10 km/h é rara na base. O detector de anomalias, operando de forma independente, confirma a atipicidade com score de -0,079. O sistema especialista consolida as duas saídas e eleva o risco a **CRÍTICO**, acionando quatro ações e listando oito justificativas rastreáveis, cada uma vinculada à condição que a disparou.
 
![Figura 10 — Incidente operacional com anomalia detectada](docs/figuras/figura_10_cenario_incidente.png)
 
### 8.3. Cenário 3 — Análise em lote
 
No modo em lote, o arquivo `dados/base_original.csv` é carregado integralmente e os 1.500 registros são analisados de uma vez, retornando estado previsto, confiança, indicação de anomalia, score, nível de risco e ações para cada linha, com opção de exportar o resultado em `.csv`.
 
![Figura 11 — Análise em lote](docs/figuras/figura_11_cenario_lote.png)
 
A tabela evidencia o funcionamento conjunto das camadas linha a linha. O registro 8, por exemplo, é classificado como INCIDENTE OPERACIONAL com 100% de confiança e simultaneamente sinalizado como anômalo (score -0,017): a concordância entre as duas camadas independentes eleva o risco a CRÍTICO. Já os registros 0 e 1, também classificados como incidente e com confiança alta, não são anômalos — correspondem a situações graves porém recorrentes na base — e recebem risco ALTO. A distinção entre risco ALTO e CRÍTICO dentro de uma mesma classe prevista é produzida inteiramente pela camada de detecção de anomalias.
 
---
 
## 9. Análise crítica dos resultados
 
### 9.1. Sobre o patamar das métricas
 
O F1-macro de 0,919 obtido no conjunto de teste situa-se em intervalo que merece comentário explícito. Um resultado próximo de 99% ou 100% em base sintética seria indício de problema, não de qualidade: significaria que as regras de geração produziram classes linearmente separáveis e que o modelo apenas reaprendeu a função determinística usada para rotular os dados, sem qualquer capacidade de generalização.
 
O patamar observado é consequência direta de duas decisões de construção da base. O ruído de rótulo introduz taxa irredutível de erro, à qual nenhum modelo pode escapar, pois parte dos rótulos é inconsistente com as variáveis por construção. E a sobreposição deliberada entre LENTIDÃO MODERADA e CONGESTIONAMENTO CRÍTICO reproduz ambiguidade que existe no fenômeno real. A concentração de 11 dos 15 erros exatamente nessa fronteira, prevista pela análise exploratória antes do treinamento, confirma que o erro residual do modelo não é aleatório: ele mapeia a incerteza genuína do problema.
 
### 9.2. Sobre o desempenho por classe
 
A leitura por classe revela um desequilíbrio que a média macro esconde. O recall no INCIDENTE OPERACIONAL é de 0,982, enquanto no CONGESTIONAMENTO CRÍTICO é de 0,806 — diferença de quase 18 pontos. Duas causas se somam: o congestionamento é a classe com menos registros na base (207, ou 13,8%) e ocupa a região de fronteira mais ambígua, enquanto o incidente dispõe de uma assinatura discriminante clara nas frenagens bruscas.
 
Do ponto de vista operacional, esse desequilíbrio é o mais favorável possível dentre os arranjos disponíveis. Se fosse necessário escolher qual classe o modelo deveria detectar melhor, a escolha recairia sobre o incidente — cuja omissão significa não acionar equipe de campo diante de um acidente. O congestionamento não detectado gera atraso e desconforto; o incidente não detectado pode gerar risco à segurança. O modelo acerta onde mais importa.
 
### 9.3. Sobre o comportamento das camadas em conjunto
 
A evidência mais forte a favor da arquitetura em camadas está no cruzamento entre a matriz de confusão e a tabela de anomalias. O erro operacionalmente mais grave do classificador — um incidente real previsto como congestionamento — ocorreu uma vez no conjunto de teste. Considerando que 17,3% dos registros de INCIDENTE OPERACIONAL são sinalizados pelo detector de anomalias, existe probabilidade não desprezível de que esse registro tenha sido capturado pela segunda camada, elevando o nível de risco e recomendando validação humana mesmo com a classificação incorreta. As duas camadas erram de formas diferentes, o que é exatamente a propriedade que se busca ao compô-las.
 
Igualmente relevante é a taxa nula de falso alarme do detector nos cenários normal e pico. Uma camada de anomalia que disparasse a cada horário de pico seria rapidamente ignorada pelos operadores — o problema clássico de fadiga de alerta em centros de monitoramento. O comportamento observado indica que o detector aprendeu a distinguir valor extremo de combinação atípica, distinção que sustenta a confiança operacional na ferramenta.
 
### 9.4. Sobre a explicabilidade da decisão
 
A saída do sistema é explicável em três níveis. No nível da **classificação**, as probabilidades por classe informam o grau de confiança e permitem identificar situações de fronteira, nas quais duas classes recebem probabilidades próximas — exatamente os casos em que se concentram os erros. No nível da **anomalia**, a indicação binária e o score informam se a situação é reconhecível frente ao histórico. No nível da **decisão**, a lista de justificativas expõe quais condições acionaram cada ação recomendada.
 
Esse encadeamento evita o cenário indesejável em que o sistema emite recomendação que o operador não tem meios de avaliar — situação que, na prática, leva ao abandono da ferramenta ou à sua aceitação acrítica, ambos resultados ruins.
 
---
 
## 10. Limitações da solução
 
As limitações abaixo são apresentadas explicitamente por delimitarem o alcance real da solução e as condições necessárias para sua eventual aplicação em produção.
 
| Limitação | Descrição e impacto |
|---|---|
| Base sintética | Os dados são gerados por regras definidas pelo autor. Ainda que respeitem relações físicas conhecidas do tráfego, não capturam a variabilidade, os ruídos de sensor e os padrões locais de dados reais. As métricas obtidas não podem ser transferidas diretamente a um cenário real sem novo treinamento e nova validação. |
| Volume reduzido de dados | Com 1.500 registros e apenas 207 de CONGESTIONAMENTO CRÍTICO, a classe minoritária dispõe de amostra pequena — provável causa do recall de 0,806. É também a limitação que penaliza a MLP frente aos modelos de árvore. |
| Ausência de dimensão espacial | Cada registro descreve um corredor isolado, sem representação da malha viária. O sistema não modela a propagação do congestionamento entre vias adjacentes nem calcula rotas alternativas concretas — apenas recomenda que sejam adotadas. |
| Ausência de dimensão temporal | Os registros são tratados como independentes, sem série temporal. O sistema classifica o estado presente, mas não projeta sua evolução nos minutos seguintes, capacidade que seria valiosa para operação preventiva. |
| Limiares fixos no sistema especialista | As regras usam limiares definidos por critérios de engenharia, não calibrados com dados operacionais reais. Sua transposição para outro corredor ou outra cidade exigiria revisão por especialista de domínio. |
| Contaminação fixa no detector | O parâmetro de 5% é uma premissa, não uma estimativa derivada dos dados. Em operação real, a proporção de eventos atípicos varia por corredor, faixa horária e sazonalidade. |
| Vazamento leve na imputação | A mediana usada para preencher os 144 valores ausentes é calculada sobre a base completa, antes do split. O correto seria ajustá-la apenas no treino, dentro de um `Pipeline`. O efeito é pequeno pelo volume envolvido e pela robustez da mediana, mas é uma imprecisão metodológica que afeta marginalmente as métricas relatadas. |
| Validação sem dados independentes | A avaliação usou partições da mesma base sintética. Não há conjunto de validação de fonte independente, o que limita a estimativa de generalização. |
 
### 10.1. Melhorias futuras
 
- Substituir ou complementar a base sintética com dados reais de sensores e sistemas de monitoramento, mantendo a arquitetura e refazendo o treinamento.
- Ampliar a amostra da classe CONGESTIONAMENTO CRÍTICO, ou aplicar reamostragem e pesos de classe, atacando diretamente o recall de 0,806.
- Incorporar modelagem temporal, permitindo previsão do estado operacional em horizonte curto em vez de apenas classificação do instante presente.
- Representar a malha viária como grafo, viabilizando cálculo efetivo de rotas alternativas por busca informada.
- Substituir os limiares fixos por lógica fuzzy calibrada com dados reais, tratando de forma mais adequada a transição entre lentidão e congestionamento, onde se concentram os erros.
- Encapsular imputação, codificação e padronização em um `Pipeline` do scikit-learn ajustado apenas sobre o treino, eliminando o vazamento residual da imputação pela mediana.
- Implementar retreinamento periódico com monitoramento de deriva, dado que padrões de tráfego mudam com obras, mudanças de uso do solo e alterações de política viária.
---
 
## 11. Conclusão
 
O sistema desenvolvido cumpre o objetivo de apoiar decisões operacionais em um centro de mobilidade urbana, indo além do treinamento de um modelo de classificação. A arquitetura em três camadas — classificação supervisionada, detecção de anomalias não supervisionada e sistema especialista — produz saída simultaneamente acionável e auditável, condição necessária em ambiente crítico no qual a decisão final permanece com o operador humano.
 
Os resultados sustentam essa escolha de arquitetura. O Random Forest atingiu 0,919 de F1-macro no teste, com recall de 0,982 na classe operacionalmente mais crítica. O detector de anomalias, treinado sem acesso aos rótulos, concentrou 88% de suas detecções nessa mesma classe e não produziu nenhum falso alarme nos 900 registros de operação de rotina. E o sistema especialista corrige uma limitação identificada no classificador, tratando como regra explícita variáveis raras que o modelo estatístico subpondera.
 
O trabalho evidencia que o valor de uma solução de inteligência artificial aplicada a operações não se concentra na métrica do modelo, mas na forma como seu resultado é integrado ao processo de decisão. Um classificador com F1-macro de 0,919 entrega pouco valor isolado; o mesmo classificador, combinado a uma verificação independente de plausibilidade e a uma camada de tradução em ação justificada, entrega uma ferramenta que um operador pode efetivamente utilizar — e questionar quando necessário.
 
---
 
## Referências
 
- BREIMAN, L. Random Forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.
- LIU, F. T.; TING, K. M.; ZHOU, Z.-H. Isolation Forest. In: IEEE INTERNATIONAL CONFERENCE ON DATA MINING, 8., 2008, Pisa. *Proceedings* [...]. Pisa: IEEE, 2008. p. 413–422.
- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011.
- McKINNEY, W. Data Structures for Statistical Computing in Python. In: PYTHON IN SCIENCE CONFERENCE, 9., 2010, Austin. *Proceedings* [...]. Austin: SciPy, 2010. p. 56–61.
- STREAMLIT. *Streamlit Documentation*. Disponível em: <https://docs.streamlit.io>. Acesso em: 3 ago. 2026.
