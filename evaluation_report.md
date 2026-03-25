# Evaluation Report — Cronologia delle Run

> Generato automaticamente dall'OrchestratorAgent con Microsoft Agent Framework.

---
## Run 1  —  2026-03-25 16:21:02

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
## Run 2  —  2026-03-25 16:23:27

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.7663** ± 0.0034 |
| Precision (CV-5) | 0.7735 |
| Recall (CV-5) | 0.7637 |
| AUC-ROC (CV-5) | 0.8496 |
| Δ vs run precedente | +0.0107  ▲ |
| Numero feature in input | 25 |

### Top correlazioni con il target (Pearson)
  - `debt_capacity_gap`: +0.4944
  - `debt_burden_ratio`: +0.4188
  - `tot_outstanding_debt`: +0.3847
  - `active_line_delinquency_burden`: +0.3778
  - `debt_pressure_index`: +0.3595
  - `annual_income`: -0.3272
  - `credit_stress_index`: +0.0335
  - `delinquency_30d_freq`: +0.0221

### Feature importance (top 10)
  - `num__debt_burden_ratio`: 0.1857
  - `num__debt_capacity_gap`: 0.1593
  - `num__debt_pressure_index`: 0.1315
  - `num__tot_outstanding_debt`: 0.0722
  - `num__active_line_delinquency_burden`: 0.0669
  - `num__annual_income`: 0.0587
  - `num__recent_customer_service_load`: 0.0357
  - `cat__account_manager_id`: 0.0354
  - `num__service_intensity_per_tenure`: 0.0340
  - `num__tenure_adjusted_relationship_frict`: 0.0305

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
1) Rischio di sovraindebitamento: costruire un indice di leva finanziaria combinando tot_outstanding_debt con annual_income per rappresentare quanto il cliente è esposto rispetto alla propria capacità reddituale; questo cattura la sostenibilità del debito, uno dei driver più forti del default. 2) Pressione creditizia e fragilità di comportamento: incrociare delinquency_30d_freq con credit_lines_count per distinguere clienti con molte linee attive ma già segnali di ritardo, evidenziando una possibile gestione stressata del credito. 3) Vitalità della relazione cliente-banca: aggregare customer_tenure_months, support_tickets_count e marketing_email_opens in un indicatore di engagement/attrito, dove molti ticket e poche aperture email su lunga anzianità possono segnalare deterioramento della relazione o minor attenzione alle comunicazioni. 4) Rischio contestuale per segmento operativo: creare feature di cross tra industry_sector e variabili finanziarie/comportamentali per catturare differenze strutturali tra settori (es. settori più ciclici con stessa leva possono avere rischio diverso), e analogamente tra branch_code/account_manager_id e segnali di morosità per intercettare effetti di filiale o di gestione del portafoglio. 5) Intensità di servizio rispetto alla maturità del cliente: combinare support_tickets_count e marketing_email_opens con customer_tenure_months per misurare se un cliente recente o storico mostra un livello di interazione anomalo rispetto al ciclo di vita, utile come segnale indiretto di frizione, difficoltà operative o scarso coinvolgimento.

---
## Run 3  —  2026-03-25 16:24:23

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.7598** ± 0.0043 |
| Precision (CV-5) | 0.7633 |
| Recall (CV-5) | 0.7580 |
| AUC-ROC (CV-5) | 0.8440 |
| Δ vs run precedente | -0.0065  ▼ |
| Numero feature in input | 25 |

### Top correlazioni con il target (Pearson)
  - `debt_capacity_gap`: +0.4944
  - `debt_burden_ratio`: +0.4188
  - `tot_outstanding_debt`: +0.3847
  - `active_line_delinquency_burden`: +0.3778
  - `debt_pressure_index`: +0.3595
  - `annual_income`: -0.3272
  - `credit_stress_index`: +0.0335
  - `delinquency_30d_freq`: +0.0221

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
L’iterazione migliore con Random Forest indica che il problema beneficia di interazioni non lineari e di segmentazione del rischio, più che di relazioni puramente lineari. La strategia aggiornata deve quindi rafforzare tre assi: 1) leva finanziaria e capacità di rimborso, costruendo segnali che combinano debito, reddito e intensità delle linee di credito per distinguere clienti semplicemente indebitati da clienti realmente sotto stress; 2) deterioramento comportamentale e relazione con la banca, integrando morosità, ticket di supporto e tenure per intercettare clienti che mostrano frizione operativa o preavvisi di default; 3) rischio contestuale di segmento, sfruttando industria, filiale e account manager per catturare differenze di portafoglio, policy o qualità del servicing. In ottica di multicollinearità, conviene preferire feature composite semanticamente robuste invece di molte variabili finanziarie ridondanti. Il modello best-fit rimane un ensemble ad albero, in particolare Random Forest o boosted tree, perché riescono a modellare soglie e interazioni tra rischio finanziario, comportamento e contesto commerciale.

---
## Run 4  —  2026-03-25 16:25:17

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.7598** ± 0.0043 |
| Precision (CV-5) | 0.7633 |
| Recall (CV-5) | 0.7580 |
| AUC-ROC (CV-5) | 0.8440 |
| Δ vs run precedente | +0.0000  ▲ |
| Numero feature in input | 25 |

### Top correlazioni con il target (Pearson)
  - `debt_capacity_gap`: +0.4944
  - `debt_burden_ratio`: +0.4188
  - `tot_outstanding_debt`: +0.3847
  - `active_line_delinquency_burden`: +0.3778
  - `debt_pressure_index`: +0.3595
  - `annual_income`: -0.3272
  - `credit_stress_index`: +0.0335
  - `delinquency_30d_freq`: +0.0221

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
Le iterazioni mostrano che le feature legate al peso del debito e alla pressione sulle linee attive sono le più informative, ma l’ultima configurazione ha perso performance perché ha probabilmente introdotto segnali troppo sovrapposti e poco diversificati. La strategia aggiornata deve quindi mantenere il focus sulla vulnerabilità finanziaria, ma arricchirlo con due dimensioni più distintive: 1) sostenibilità del debito nel tempo, combinando debito, reddito e tenure per distinguere un indebitamento fisiologico da uno strutturalmente rischioso; 2) qualità del comportamento creditizio, integrando morosità con l’ampiezza dell’esposizione per misurare la pressione reale sulle linee attive; 3) stato della relazione cliente-banca, usando supporto e marketing come segnali di frizione, disengagement o difficoltà operative; 4) segmentazione contestuale per settore/filiale/manager per intercettare effetti di portafoglio e qualità del servicing. In sintesi, conviene passare da soli ratio di debito a indicatori compositi che riflettano rischio finanziario + maturità della relazione + contesto operativo, riducendo ridondanza tra variabili strettamente collegate.

---
## Run 5  —  2026-03-25 16:26:09

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.7598** ± 0.0043 |
| Precision (CV-5) | 0.7633 |
| Recall (CV-5) | 0.7580 |
| AUC-ROC (CV-5) | 0.8440 |
| Δ vs run precedente | +0.0000  ▲ |
| Numero feature in input | 25 |

### Top correlazioni con il target (Pearson)
  - `debt_capacity_gap`: +0.4944
  - `debt_burden_ratio`: +0.4188
  - `tot_outstanding_debt`: +0.3847
  - `active_line_delinquency_burden`: +0.3778
  - `debt_pressure_index`: +0.3595
  - `annual_income`: -0.3272
  - `credit_stress_index`: +0.0335
  - `delinquency_30d_freq`: +0.0221

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
Le iterazioni precedenti indicano che il segnale più forte viene dalla dimensione di pressione finanziaria, in particolare dal debito rapportato alla capacità e dalla morosità sulle linee attive. Tuttavia, le feature già provate risultano molto concentrate sullo stesso fenomeno, quindi la strategia va aggiornata per mantenere il nucleo di rischio creditizio ma aggiungere una lettura più sfumata del ciclo di vita e del comportamento relazionale. In pratica: 1) confermare la centralità dell’indebitamento sostenibile, ma trasformarlo in indicatori che distinguano esposizione assoluta, carico relativo e resilienza nel tempo; 2) arricchire il segnale di morosità con una lettura di “intensità del problema” rispetto alla relazione e al numero di linee, così da separare incidenti occasionali da difficoltà strutturali; 3) introdurre feature di engagement/attrito per catturare clienti che interagiscono poco o hanno bisogno di più supporto, perché questo può anticipare deterioramento; 4) preservare il contesto di filiale, manager e settore solo come layer di segmentazione, non come segnale dominante. La direzione corretta è quindi passare da ratio finanziari isolati a indici compositi di rischio economico, stress operativo e qualità della relazione, mantenendo un modello ad alberi per sfruttare interazioni e soglie.
