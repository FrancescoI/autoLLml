
---
## Run 1  —  2026-03-18 15:47:44

### Metriche
| Metrica | Valore |
|---------|--------|
| R2 Mean (CV-5) | **0.3902** ± 0.0381 |
| Δ vs run precedente | — (baseline) |
| Numero feature in input | 5 |

### Top correlazioni con il target (Pearson)
  - `delinquency_30d_freq`: +0.5113
  - `tot_outstanding_debt`: +0.5097
  - `credit_lines_count`: +0.3699
  - `annual_income`: +0.2177

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
*(non generata — run baseline)*

---
## Run 1  —  2026-03-18 15:56:06

### Metriche
| Metrica | Valore |
|---------|--------|
| R2 Mean (CV-5) | **0.3902** ± 0.0381 |
| Δ vs run precedente | — (baseline) |
| Numero feature in input | 5 |

### Top correlazioni con il target (Pearson)
  - `delinquency_30d_freq`: +0.5113
  - `tot_outstanding_debt`: +0.5097
  - `credit_lines_count`: +0.3699
  - `annual_income`: +0.2177

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
*(non generata — run baseline)*

---
## Run 1  —  2026-03-18 15:56:36

### Metriche
| Metrica | Valore |
|---------|--------|
| R2 Mean (CV-5) | **0.3902** ± 0.0381 |
| Δ vs run precedente | — (baseline) |
| Numero feature in input | 5 |

### Top correlazioni con il target (Pearson)
  - `delinquency_30d_freq`: +0.5113
  - `tot_outstanding_debt`: +0.5097
  - `credit_lines_count`: +0.3699
  - `annual_income`: +0.2177

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
*(non generata — run baseline)*

---
## Run 1  —  2026-03-18 15:57:31

### Metriche
| Metrica | Valore |
|---------|--------|
| R2 Mean (CV-5) | **0.3902** ± 0.0381 |
| Δ vs run precedente | — (baseline) |
| Numero feature in input | 5 |

### Top correlazioni con il target (Pearson)
  - `delinquency_30d_freq`: +0.5113
  - `tot_outstanding_debt`: +0.5097
  - `credit_lines_count`: +0.3699
  - `annual_income`: +0.2177

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
*(non generata — run baseline)*

---
## Run 1  —  2026-03-18 16:02:33

### Metriche
| Metrica | Valore |
|---------|--------|
| R2 Mean (CV-5) | **0.3553** ± 0.0184 |
| Δ vs run precedente | — (baseline) |
| Numero feature in input | 30 |

### Top correlazioni con il target (Pearson)
  - `debt_to_income`: +0.5925
  - `is_debt_bucket__low`: -0.5181
  - `delinquency_30d_freq`: +0.5113
  - `default_risk_by_sector_interaction`: +0.5113
  - `tot_outstanding_debt`: +0.5097
  - `high_delinquency`: +0.4962
  - `is_debt_bucket__high`: +0.4855
  - `annual_income_to_debt`: -0.4831
  - `risk_score_base`: +0.4607
  - `high_debt_to_income`: +0.4604

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
*(non generata — run baseline)*

---
## Run 1  —  2026-03-18 16:19:47

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.8409** ± 0.0074 |
| Δ vs run precedente | — (baseline) |
| Numero feature in input | 5 |

### Top correlazioni con il target (Pearson)
  - `delinquency_30d_freq`: +0.5113
  - `tot_outstanding_debt`: +0.5097
  - `credit_lines_count`: +0.3699
  - `annual_income`: +0.2177

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
*(non generata — run baseline)*

---
## Run 2  —  2026-03-18 16:21:10

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.8400** ± 0.0059 |
| Δ vs run precedente | -0.0009  ▼ |
| Numero feature in input | 22 |

### Top correlazioni con il target (Pearson)
  - `debt_to_income`: +0.5552
  - `flag_dti_ge_1p5`: +0.5524
  - `delinquency_30d_freq`: +0.5113
  - `tot_outstanding_debt`: +0.5097
  - `flag_delinq_ge_2`: +0.4962
  - `lines_x_delinq`: +0.4754
  - `complexity_score`: +0.4527
  - `flag_delinq_pos`: +0.4397
  - `debt_per_line`: +0.4096
  - `delinq_x_debt`: +0.3804

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
1) Stress/Leverage ratios: creare feature ratio come tot_outstanding_debt / annual_income (debt-to-income), delinquency_30d_freq * tot_outstanding_debt (rischio ponderato) e tot_outstanding_debt / max(1, credit_lines_count) (debito per linea). Queste trasformano il profilo finanziario in segnali direttamente interpretabili per il pricing e la concessione.
2) Integrazione morosità + esposizione: combinare delinquency_30d_freq con la dimensione dell’esposizione via crossing (es. buckets di delinquency_30d_freq × classi di debt-to-income) e creare aggregazioni indicative come (credit_lines_count * delinquency_30d_freq) per catturare instabilità con più linee attive.
3) Segmentazione per settore con aggregazioni: creare statistiche target-encoding “safe” per industry_sector (es. default rate storico per settore, con smoothing) e cross settore × debt-to-income bucket per individuare differenze di rischio strutturali.
4) Interazioni di capacità/complessità: crossing tra credit_lines_count e debt-to-income (es. alta complessità: molte linee + elevato leverage; oppure molte linee ma basso leverage) per discriminare clienti con apparente capacità ma comportamento sotto stress.
5) Feature di soglia e ranking semantico: costruire indicatori basati su soglie business (es. debt-to-income sopra soglia A/B; delinquency_30d_freq > 0; credit_lines_count >= k) e usarli in combinazione con industry_sector. Riduce il rumore rispetto a trasformazioni continue aggressive e supporta policy di underwriting.
