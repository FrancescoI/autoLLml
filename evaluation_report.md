# Evaluation Report — Cronologia delle Run

> Generato automaticamente dall'OrchestratorAgent con Microsoft Agent Framework.

---
## Run 1  —  2026-03-25 16:54:57

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.7556** ± 0.0098 |
| Precision (CV-5) | 0.7570 |
| Recall (CV-5) | 0.7608 |
| AUC-ROC (CV-5) | 0.8413 |
| Δ vs run precedente | — (baseline) |
| Numero feature in input | 10 |

### Top correlazioni con il target (Pearson)
  - `tot_outstanding_debt`: +0.3847
  - `annual_income`: -0.3272
  - `credit_lines_count`: +0.0524
  - `customer_tenure_months`: +0.0242
  - `delinquency_30d_freq`: +0.0221
  - `marketing_email_opens`: -0.0080
  - `support_tickets_count`: +0.0045

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
*(non generata — run baseline)*

---
## Run 2  —  2026-03-25 16:56:00

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.7752** ± 0.0016 |
| Precision (CV-5) | 0.7802 |
| Recall (CV-5) | 0.7731 |
| AUC-ROC (CV-5) | 0.8572 |
| Δ vs run precedente | +0.0196  ▲ |
| Numero feature in input | 27 |

### Top correlazioni con il target (Pearson)
  - `income_debt_balance_bucket`: -0.5463
  - `debt_sustainability_index`: -0.5054
  - `debt_burden_gap`: +0.5054
  - `multi_dim_vulnerability`: +0.4557
  - `tot_outstanding_debt`: +0.3847
  - `annual_income`: -0.3272

### Feature importance (top 10)
  - `num__debt_burden_gap`: 0.4128
  - `num__debt_sustainability_index`: 0.4009
  - `num__annual_income`: 0.0551
  - `num__tot_outstanding_debt`: 0.0525
  - `num__credit_lines_count`: 0.0107
  - `num__multi_dim_vulnerability`: 0.0094
  - `num__manager_historical_default_rate`: 0.0088
  - `num__low_engagement_high_service`: 0.0056
  - `num__customer_tenure_months`: 0.0051
  - `num__behavioral_credit_stress`: 0.0050

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
1) Indice di sostenibilità del debito: combinare reddito annuo e debito residuo per rappresentare la capacità del cliente di assorbire il servizio del debito; utile perché il default è spesso guidato dal disallineamento tra flussi in entrata e esposizione finanziaria. 2) Stress creditizio comportamentale: incrociare frequenza di morosità a 30 giorni con numero di linee di credito attive per catturare clienti con segni di tensione su portafogli più complessi; questa dinamica riflette l’aumento del rischio quando più linee vengono gestite con ritardi ricorrenti. 3) Stabilità della relazione cliente-banca: combinare customer_tenure_months con support_tickets_count e marketing_email_opens per distinguere clienti consolidati e ingaggiati da clienti “fragili” con molte richieste di assistenza e bassa risposta ai contatti; segnali di frizione operativa possono precedere il default. 4) Rischio di contesto operativo/commerciale: aggregare industry_sector, branch_code e account_manager_id per creare indicatori di rischio relativo per segmento/filiale/gestore, catturando differenze strutturali di portafoglio e qualità della gestione; questo è utile per identificare sacche di rischio non spiegate dalle sole variabili cliente. 5) Profilo di vulnerabilità multi-dimensionale: costruire crossing tra settore industriale e metriche finanziarie/comportamentali (es. industria ad alta volatilità + alta morosità + debito elevato) per intercettare combinazioni di business più esposte a shock di liquidità e default.

---
## Run 3  —  2026-03-25 16:57:05

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.7743** ± 0.0018 |
| Precision (CV-5) | 0.7796 |
| Recall (CV-5) | 0.7721 |
| AUC-ROC (CV-5) | 0.8561 |
| Δ vs run precedente | -0.0009  ▼ |
| Numero feature in input | 34 |

### Top correlazioni con il target (Pearson)
  - `high_leverage_pressure`: +0.5054
  - `debt_sustainability_index`: -0.5054
  - `debt_burden_gap`: +0.5054
  - `debt_credit_complexity`: +0.3847
  - `tot_outstanding_debt`: +0.3847
  - `multi_dim_vulnerability`: +0.3439

### Feature importance (top 10)
  - `num__high_leverage_pressure`: 0.3796
  - `num__debt_sustainability_index`: 0.3276
  - `num__debt_burden_gap`: 0.1122
  - `num__annual_income`: 0.0567
  - `num__tot_outstanding_debt`: 0.0256
  - `num__debt_credit_complexity`: 0.0235
  - `num__credit_lines_count`: 0.0083
  - `num__financial_behavioral_risk_bundle`: 0.0082
  - `cat__manager_friction_profile`: 0.0054
  - `num__industry_manager_group_size`: 0.0050

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
La performance è stagnante: la migliore iterazione è rimasta ferma a 0.7752, quindi non emerge un miglioramento rispetto all’ultima prova. Va quindi rafforzata la strategia con feature che catturino segnali di rischio più vicini al comportamento reale del cliente, non solo la sua fotografia finanziaria. Le opportunità più promettenti sono: 1) intensità di stress finanziario, cioè il rapporto tra debito, reddito e complessità del credito per distinguere clienti con leva eccessiva; 2) segnali di deterioramento comportamentale, combinando morosità, ticket di supporto e tenure per intercettare clienti che stanno entrando in difficoltà operativa e relazionale; 3) rischio di portafoglio/contesto, usando aggregazioni per settore, filiale e account manager per identificare aree o gestioni con tassi di default strutturalmente più alti; 4) engagement vs frizione, confrontando aperture marketing e ticket di supporto per capire se il cliente è ingaggiato o solo reattivo ai problemi. La strategia precedente va confermata ma ampliata con crossing più informativi tra finanza, relazione e contesto operativo.

---
## Run 4  —  2026-03-25 16:58:07

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.7754** ± 0.0023 |
| Precision (CV-5) | 0.7807 |
| Recall (CV-5) | 0.7732 |
| AUC-ROC (CV-5) | 0.8565 |
| Δ vs run precedente | +0.0011  ▲ |
| Numero feature in input | 35 |

### Top correlazioni con il target (Pearson)
  - `sustainable_leverage_flag`: -0.5551
  - `financial_stress_normalized`: +0.5054
  - `debt_sustainability_index`: -0.5054
  - `debt_burden_gap`: +0.5054
  - `multi_dim_risk_profile`: +0.3939
  - `credit_spread_pressure`: +0.3847

### Feature importance (top 10)
  - `num__debt_sustainability_index`: 0.3798
  - `num__financial_stress_normalized`: 0.2573
  - `num__debt_burden_gap`: 0.1795
  - `num__annual_income`: 0.0560
  - `num__tot_outstanding_debt`: 0.0291
  - `num__credit_spread_pressure`: 0.0227
  - `num__credit_lines_count`: 0.0131
  - `num__early_fragility_signal`: 0.0092
  - `num__multi_dim_risk_profile`: 0.0071
  - `cat__manager_fragility_profile`: 0.0055

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
La tendenza è in peggioramento (0.7752 -> 0.7743), quindi la sola raffinazione dell’indice di sostenibilità del debito non sta aggiungendo informazione sufficiente. Bisogna cambiare enfasi: non solo capacità finanziaria statica, ma combinazioni di rischio che descrivono la vulnerabilità operativa e la qualità della relazione con la banca. Le aree più promettenti sono: 1) stress finanziario normalizzato, che metta in relazione debito, reddito e numero di linee per distinguere leve sostenibili da profili sovraesposti; 2) deterioramento relazionale, dove ticket di supporto elevati e bassa interazione marketing indicano attrito, possibile insoddisfazione o difficoltà di gestione; 3) rischio di contesto multi-livello, con aggregazioni per settore, filiale e account manager per catturare differenze strutturali di portafoglio e di qualità di servizio; 4) segnali di fragilità precoce, combinando delinquency_30d_freq con customer_tenure_months per distinguere nuovi clienti a rischio da clienti storicamente stabili. La strategia precedente va mantenuta come base, ma va arricchita con indicatori relazionali e di portafoglio più vicini ai driver reali del default.

---
## Run 5  —  2026-03-25 16:59:05

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.7747** ± 0.0032 |
| Precision (CV-5) | 0.7799 |
| Recall (CV-5) | 0.7725 |
| AUC-ROC (CV-5) | 0.8570 |
| Δ vs run precedente | -0.0007  ▼ |
| Numero feature in input | 33 |

### Top correlazioni con il target (Pearson)
  - `financial_stress_normalized`: +0.5054
  - `debt_sustainability_index`: -0.5054
  - `debt_burden_gap`: +0.5054
  - `composite_risk_core`: +0.4108
  - `leverage_complexity_pressure`: +0.3847
  - `tot_outstanding_debt`: +0.3847

### Feature importance (top 10)
  - `num__financial_stress_normalized`: 0.3744
  - `num__debt_sustainability_index`: 0.3243
  - `num__debt_burden_gap`: 0.1192
  - `num__annual_income`: 0.0545
  - `num__leverage_complexity_pressure`: 0.0256
  - `num__tot_outstanding_debt`: 0.0227
  - `num__multi_line_pressure`: 0.0142
  - `num__composite_risk_core`: 0.0115
  - `num__credit_lines_count`: 0.0065
  - `cat__manager_fragility_profile`: 0.0056

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
La tendenza è in plateau, con un lieve miglioramento nell’ultima iterazione (0.7752 -> 0.7743 -> 0.7754), quindi la strategia base sul rischio di indebitamento continua a funzionare ma ha raggiunto un tetto. I pattern ricorrenti indicano che le variabili di debito e reddito restano il nucleo predittivo, ma manca ancora una lettura più “aziendale” del deterioramento. Conviene quindi consolidare il blocco finanziario e aggiungere tre nuove dimensioni: 1) pressione da leva multi-linea, per misurare quando il debito è sostenibile solo in presenza di poche linee ma diventa fragile all’aumentare della complessità creditizia; 2) segnali di disallineamento relazione-cliente, dove alta attività di supporto e bassa risposta marketing suggeriscono deterioramento operativo o scarso engagement; 3) rischio di ecosistema commerciale, cioè differenze sistematiche tra settore, filiale e gestore che possono riflettere qualità del portafoglio o del presidio commerciale. In sintesi, mantenere l’asse debt sustainability ma arricchirlo con feature che catturano fragilità comportamentale e rischio di contesto per superare il plateau.
