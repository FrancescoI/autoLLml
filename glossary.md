# Target Variable
- `consumo_annuo`: Consumo di gas annuale di un'azienda in metri cubi.

# Feature Dictionary
- `p_iva_finale`: Partita_Iva dell'azienda (es: 8115360961)
- `DENOMINAZIONE`: Denominazione dell'azienda (es: SURGY-MEDICA S.R.L.)
- `FLAG_GRUPPO`: Flag che indica se l'azienda fa parte di un gruppo (1 = fa parte di un gruppo)
- `FORMA_GIURIDICA`: Forma giuridica dell'azienda (es: S.R.L.)
- `Provincia2019`: Provincia in cui si trova la sede legale dell'azienda (es: MI)
- `Comune2019`: Comune in cui si trova la sede legale dell'azienda (es: MILANO)
- `regione2019`: Regione in cui si trova la sede legale dell'azienda (es: Lombardia)
- `ateco`: Codice Ateco 2007 (es: 477110)
- `_12_settori_cra`: Settore di appartenenza dell'azienda, come da sistema di riclassificazione della rating agency di CRIF. Suddivisione in 12 settori
- `_25_settori_cra`: Settore di appartenenza dell'azienda, come da sistema di riclassificazione della rating agency di CRIF. Suddivisione in 25 settori
- `_118_settori_cra`: Settore di appartenenza dell'azienda, come da sistema di riclassificazione della rating agency di CRIF. Suddivisione in 118 settori, codice identificativo
- `label_118_microsettori`: Settore di appartenenza dell'azienda, come da sistema di riclassificazione della rating agency di CRIF. Suddivisione in 118 settori, etichetta descrittiva
- `CASHFLOW`: Cashflow dell'azienda (es: 1000000)
- `CREDBREVE`: Variabile numerica non chiara, da approfondire
- `IMMMATER`: Immobilizzazione immateriali (es: 1000000)
- `MOL`: Margine Operativo Lordo (es: 1000000)
- `PN`: Variabile numerica non chiara, da approfondire
- `REDDOPE`: Reddito Operativo (es: 1000000)
- `RICAVI`: Ricavi (es: 1000000)
- `RICAVITOT`: Ricavi Totali (es: 1000000)
- `TOTATTIVO`: Totale Attivo (es: 1000000)
- `TOTDEB`: Totale Debiti (es: 1000000)
- `UTILEPERDITA`: Utile/Perdita (es: 1000000)
- `VALPROD`: Valore Produzione (es: 1000000)
- `ACQUISTI`: Totale Acquisti (es: 1000000)
- `CAPSOC`: Capitale Sociale (es: 1000000)
- `CONSUMIMATERIE`: Costo consumi materie prime (es: 1000000)
- `SPESEGESTIONALI`: Spese gestionali (es: 1000000)
- `TOTPASSIVO`: Totale Passivo (es: 1000000)
- `RISANTEIMP`: Risultato ante imposte (es: 1000000)
- `TOTAMMANDSVAL`: Totale ammortamenti e valutazioni (es: 1000000)
- `TOTONFIN`: Totale oneri finanziari (es: 1000000)
- `num_certificati_bianchi`: Numero di certificati bianchi (es: 1000000)
- `erogato_cip`: Erogato CIP (es: 1000000)
- `erogato_ce`: Erogato CE (es: 1000000)
- `erogato_ct`: Erogato CT (es: 1000000)
- `erogato_fer`: Erogato FER (es: 1000000)
- `erogato_omni`: Erogato OMNI (es: 1000000)
- `erogato_grin`: Erogato GRIN (es: 1000000)
- `flag_gse`: Flag GSE, se ha ricevuto un contributo in conto energia dal GSE
- `flag_ccse`: Flag CCSE, se ha ricevuto un contributo in conto energia dal CCSE
- `ateco_2`: ATECO 2007, classificazione a due valori (es: 74)
- `ateco_4`: ATECO 2007, classificazione a quattro valori (es: 7490)
- `flag_area`: flag da approfondire
- `area_nielsen`: area Nielsen di appartenenza (es: 'area1')
- `flag_num_certificati_bianchi_flag`: flag se l'azienda ha un certificato bianco


# Domain Constraints
- I valori nulli non possono essere scartati brutalmente; esplorare imputazione o target encoding per le variabili categoriche.
- Non utilizzare variabili ad alta cardinalità come p_iva_finale, DENOMINAZIONE, FORMA_GIURIDICA, Provincia, etc...
- Se necessario, riclassifica alcune variabili categorie ad alta cardinalità
- Prediligi modelli RandomForest e BoostedTree

# Obiettivo
- Devi sviluppare un modello di machine learning che sia in grado di prevedere i consumi gas di un'azienda in base alle sue caratteristiche