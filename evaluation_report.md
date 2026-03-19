# Evaluation Report — Cronologia delle Run

> Generato automaticamente dall'OrchestratorAgent con Microsoft Agent Framework.

---
## Run 1  —  2026-03-19 17:38:32

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.8483** ± 0.0096 |
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
## Run 2  —  2026-03-19 17:40:47

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.8446** ± 0.0068 |
| Δ vs run precedente | -0.0037  ▼ |
| Numero feature in input | 17 |

### Top correlazioni con il target (Pearson)
  - `ratio_outstanding_to_income`: +0.5925
  - `debt_high`: +0.5542
  - `debt_high_and_delinquent`: +0.5286
  - `delinquency_30d_freq`: +0.5113
  - `credit_utilization_style`: +0.4754
  - `delinquency_x_debt_to_income`: +0.4607
  - `many_lines_and_delinquency`: +0.4599
  - `delinquency_flag`: +0.4397

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
1) Costruisci indicatori di “capacità di rimborso” e “carico debito”: ratio_outstanding_to_income = tot_outstanding_debt / annual_income e bucket/carico (es. alto/medio/basso). Utile per distinguere clienti con lo stesso debito ma capacità di reddito diversa.
2) Feature crossing tra rischio storico e esposizione: delinquency_30d_freq x (tot_outstanding_debt / annual_income) per catturare casi in cui la morosità recente è amplificata da un carico debito elevato; includere anche un crossing discreto del tipo delinquency_flag (delinquency_30d_freq>0) + credit_lines_count (es. “molte linee + delinquency”).
3) Normalizza la complessità del portafoglio crediti: debt_per_credit_line = tot_outstanding_debt / credit_lines_count e credit_utilization_style = delinquency_30d_freq * credit_lines_count. Queste combinazioni riflettono intensità di esposizione per linea e possibile stress operativo.
4) Segmentazione tariffaria/decisionale per settore con aggregazioni target-agnostiche: creare un “sector_risk_index” basato su aggregazioni robusthe per industry_sector (es. media predittiva out-of-fold o median default rate per settore, con smoothing). Usarla come input per modelli e come regola business per pricing/limit policy.
5) Soglie business-driven su combinazioni: create feature di soglia (es. annual_income_low, debt_high) e crossing con delinquency (annual_income_low & delinquency>0; debt_high & delinquency>0). Consentono campagne di underwriting mirate (es. revisione manuale o riduzione fido) sui segmenti ad alto rischio.

---
## Run 3  —  2026-03-19 17:41:38

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.8446** ± 0.0068 |
| Δ vs run precedente | +0.0000  ▲ |
| Numero feature in input | 17 |

### Top correlazioni con il target (Pearson)
  - `ratio_outstanding_to_income`: +0.5925
  - `debt_high`: +0.5542
  - `debt_high_and_delinquent`: +0.5286
  - `delinquency_30d_freq`: +0.5113
  - `credit_utilization_style`: +0.4754
  - `delinquency_x_debt_to_income`: +0.4607
  - `many_lines_and_delinquency`: +0.4599
  - `delinquency_flag`: +0.4397

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
1) Costruisci indicatori di “capacità di rimborso” e “carico debito”: ratio_outstanding_to_income = tot_outstanding_debt / annual_income e bucket/carico (es. alto/medio/basso). Utile per distinguere clienti con lo stesso debito ma capacità di reddito diversa.
2) Feature crossing tra rischio storico e esposizione: delinquency_30d_freq x (tot_outstanding_debt / annual_income) per catturare casi in cui la morosità recente è amplificata da un carico debito elevato; includere anche un crossing discreto del tipo delinquency_flag (delinquency_30d_freq>0) + credit_lines_count (es. “molte linee + delinquency”).
3) Normalizza la complessità del portafoglio crediti: debt_per_credit_line = tot_outstanding_debt / credit_lines_count e credit_utilization_style = delinquency_30d_freq * credit_lines_count. Queste combinazioni riflettono intensità di esposizione per linea e possibile stress operativo.
4) Segmentazione tariffaria/decisionale per settore con aggregazioni target-agnostiche: creare un “sector_risk_index” basato su aggregazioni robusthe per industry_sector (es. media predittiva out-of-fold o median default rate per settore, con smoothing). Usarla come input per modelli e come regola business per pricing/limit policy.
5) Soglie business-driven su combinazioni: create feature di soglia (es. annual_income_low, debt_high) e crossing con delinquency (annual_income_low & delinquency>0; debt_high & delinquency>0). Consentono campagne di underwriting mirate (es. revisione manuale o riduzione fido) sui segmenti ad alto rischio.

---
## Run 4  —  2026-03-19 17:42:25

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.8446** ± 0.0068 |
| Δ vs run precedente | +0.0000  ▲ |
| Numero feature in input | 17 |

### Top correlazioni con il target (Pearson)
  - `ratio_outstanding_to_income`: +0.5925
  - `debt_high`: +0.5542
  - `debt_high_and_delinquent`: +0.5286
  - `delinquency_30d_freq`: +0.5113
  - `credit_utilization_style`: +0.4754
  - `delinquency_x_debt_to_income`: +0.4607
  - `many_lines_and_delinquency`: +0.4599
  - `delinquency_flag`: +0.4397

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
1) Costruisci indicatori di “capacità di rimborso” e “carico debito”: ratio_outstanding_to_income = tot_outstanding_debt / annual_income e bucket/carico (es. alto/medio/basso). Utile per distinguere clienti con lo stesso debito ma capacità di reddito diversa.
2) Feature crossing tra rischio storico e esposizione: delinquency_30d_freq x (tot_outstanding_debt / annual_income) per catturare casi in cui la morosità recente è amplificata da un carico debito elevato; includere anche un crossing discreto del tipo delinquency_flag (delinquency_30d_freq>0) + credit_lines_count (es. “molte linee + delinquency”).
3) Normalizza la complessità del portafoglio crediti: debt_per_credit_line = tot_outstanding_debt / credit_lines_count e credit_utilization_style = delinquency_30d_freq * credit_lines_count. Queste combinazioni riflettono intensità di esposizione per linea e possibile stress operativo.
4) Segmentazione tariffaria/decisionale per settore con aggregazioni target-agnostiche: creare un “sector_risk_index” basato su aggregazioni robusthe per industry_sector (es. media predittiva out-of-fold o median default rate per settore, con smoothing). Usarla come input per modelli e come regola business per pricing/limit policy.
5) Soglie business-driven su combinazioni: create feature di soglia (es. annual_income_low, debt_high) e crossing con delinquency (annual_income_low & delinquency>0; debt_high & delinquency>0). Consentono campagne di underwriting mirate (es. revisione manuale o riduzione fido) sui segmenti ad alto rischio.

---
## Run 5  —  2026-03-19 17:43:17

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.8446** ± 0.0068 |
| Δ vs run precedente | +0.0000  ▲ |
| Numero feature in input | 17 |

### Top correlazioni con il target (Pearson)
  - `ratio_outstanding_to_income`: +0.5925
  - `debt_high`: +0.5542
  - `debt_high_and_delinquent`: +0.5286
  - `delinquency_30d_freq`: +0.5113
  - `credit_utilization_style`: +0.4754
  - `delinquency_x_debt_to_income`: +0.4607
  - `many_lines_and_delinquency`: +0.4599
  - `delinquency_flag`: +0.4397

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
1) Costruisci indicatori di “capacità di rimborso” e “carico debito”: ratio_outstanding_to_income = tot_outstanding_debt / annual_income e bucket/carico (es. alto/medio/basso). Utile per distinguere clienti con lo stesso debito ma capacità di reddito diversa.
2) Feature crossing tra rischio storico e esposizione: delinquency_30d_freq x (tot_outstanding_debt / annual_income) per catturare casi in cui la morosità recente è amplificata da un carico debito elevato; includere anche un crossing discreto del tipo delinquency_flag (delinquency_30d_freq>0) + credit_lines_count (es. “molte linee + delinquency”).
3) Normalizza la complessità del portafoglio crediti: debt_per_credit_line = tot_outstanding_debt / credit_lines_count e credit_utilization_style = delinquency_30d_freq * credit_lines_count. Queste combinazioni riflettono intensità di esposizione per linea e possibile stress operativo.
4) Segmentazione tariffaria/decisionale per settore con aggregazioni target-agnostiche: creare un “sector_risk_index” basato su aggregazioni robusthe per industry_sector (es. media predittiva out-of-fold o median default rate per settore, con smoothing). Usarla come input per modelli e come regola business per pricing/limit policy.
5) Soglie business-driven su combinazioni: create feature di soglia (es. annual_income_low, debt_high) e crossing con delinquency (annual_income_low & delinquency>0; debt_high & delinquency>0). Consentono campagne di underwriting mirate (es. revisione manuale o riduzione fido) sui segmenti ad alto rischio.

---
## Run 1  —  2026-03-19 18:05:40

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.8483** ± 0.0096 |
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
## Run 2  —  2026-03-19 18:06:48

### Metriche
| Metrica | Valore |
|---------|--------|
| Task Type | CLASSIFICATION |
| F1_weighted Mean (CV-5) | **0.8452** ± 0.0029 |
| Δ vs run precedente | -0.0031  ▼ |
| Numero feature in input | 13 |

### Top correlazioni con il target (Pearson)
  - `delinquency_30d_freq`: +0.5113
  - `tot_outstanding_debt`: +0.5097
  - `delinquency_index`: +0.4754
  - `delinquency_30d_per_line`: +0.3907
  - `credit_lines_count`: +0.3699
  - `annual_income`: +0.2177
  - `settore_risk_score`: +0.0988

### Feature importance (top 10)
  *(non disponibile)*

### Feature implementate in questa run
  *(baseline)*

### Business strategy applicata
1) Costruire un indicatore di sostenibilità del debito: debt_to_income = tot_outstanding_debt / annual_income (con gestione di divisioni per valori nulli/zero). Business meaning: quanto reddito disponibile viene assorbito dal debito; è un driver diretto della probabilità di default. 2) Misurare la pressione di insolvenza per numero di linee: delinquency_index = delinquency_30d_freq * credit_lines_count (o delinquency_30d_rate_per_line). Business meaning: non solo quanta morosità emerge, ma anche su quante esposizioni/linee si distribuisce il problema. 3) Creare un indicatore “overexposure” verso il settore: settore_risk_score = aggregazione target-encoding cross-validata su industry_sector (media di default per settore calcolata su training fold, con smoothing e gestione categorie rare/Unknown). Business meaning: alcuni settori hanno dinamiche macro/creditizie diverse che influenzano il rischio. 4) Definire una combinazione rischio-coerente tra storico e struttura del credito: delinquency_weighted_debt = (delinquency_30d_freq * tot_outstanding_debt) / max(annual_income, small_value). Business meaning: debito elevato già “stressato” da morosità recente, normalizzato rispetto alla capacità di rimborso. 5) Segmentazione rischio tramite bucket semantici (senza trasformazioni arbitrarie): income_band (bassi/medi/alta capacità) × debt_band (basso/medio/alto indebitamento) come feature categorica derivata. Business meaning: la relazione rischio-reddito e rischio-debito spesso è non lineare a livello di segmenti operativi; i band aiutano i modelli tree-based a catturare policy e regimi di rischio.
