# Target Variable
- `default_flag`: Flag di default creditizio (1 = default, 0 = no default)

# Feature Dictionary
- `annual_income`: Reddito annuale del cliente (es: 50000)
- `tot_outstanding_debt`: Debito totale in essere (es: 10000)
- `credit_lines_count`: Numero di linee di credito attive (es: 3)
- `delinquency_30d_freq`: Frequenza di morosità a 30 giorni (es: 0.15)
- `industry_sector`: Settore industriale di appartenenza (es: 'Retail', 'Technology', 'Healthcare')

# Domain Constraints
- I valori nulli devono essere gestiti con imputazione (mediana per numeriche, 'Unknown' per categoriche)
- Prediligi modelli RandomForest e BoostedTree per problemi di classificazione
- Gestisci la multicollinearità tra variabili finanziarie

# Obiettivo
- Sviluppare un modello di machine learning per predire il default creditizio basato sulle caratteristiche finanziarie e demografiche del cliente
