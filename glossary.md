# Target Variable
- `default_flag`: Flag di default creditizio (1 = default, 0 = no default)

# Feature Dictionary
## Caratteristiche Finanziarie
- `annual_income`: Reddito annuale del cliente (es: 50000)
- `tot_outstanding_debt`: Debito totale in essere (es: 10000)
- `credit_lines_count`: Numero di linee di credito attive (es: 3)
- `delinquency_30d_freq`: Frequenza di morosità a 30 giorni (es: 0.15)

## Caratteristiche Categoriche
- `industry_sector`: Settore industriale di appartenenza (es: 'Retail', 'Technology', 'Healthcare')
- `branch_code`: Codice identificativo della Filiale (es: 'BR001', 'BR015')
- `account_manager_id`: ID del gestore account assegnato (es: 'AM023', 'AM047')

## Caratteristiche Categoriche
- `industry_sector`: Settore industriale di appartenenza (es: 'Retail', 'Technology', 'Healthcare')
- `branch_code`: Codice identificativo della Filiale (es: 'BR001', 'BR015')
- `account_manager_id`: ID del gestore account assegnato (es: 'AM023', 'AM047')

## Metriche di Customer Relationship
- `customer_tenure_months`: Mesi dall'onboarding del cliente
- `support_tickets_count`: Numero di ticket di supporto inviati
- `marketing_email_opens`: Aperture email marketing

# Domain Constraints
- I valori nulli devono essere gestiti con imputazione (mediana per numeriche, 'Unknown' per categoriche)
- Prediligi modelli RandomForest e BoostedTree per problemi di classificazione
- Gestisci la multicollinearità tra variabili finanziarie

# Obiettivo
- Sviluppare un modello di machine learning per predire il default creditizio basato sulle caratteristiche finanziarie e demografiche del cliente
