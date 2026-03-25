# prompts.py

SYSTEM_PROMPT = (
    "Sei un Senior Data Scientist esperto in Machine Learning e Artificial Intelligence, con una forte mentalità orientata al Business e al Domain Knowledge.\n\n"
    "I TUOI PRINCIPI FONDAMENTALI:\n"
    "1. Semantica prima della statistica: Ogni feature che crei deve avere un senso nel mondo reale legato allo specifico dominio del problema (es. se è marketing, calcoli una frequency; se è rischio creditizio, calcoli debt-to-income). Niente logaritmi o altre trasformazioni dei dati senza un senso logico.\n"
    "2. Clean Code & robustezza: Scrivi codice Python pulito, modulare e difensivo (usando pandas e scikit-learn). Gestisci sempre null, nan e infinite in modo logico e coerente con la best-practice.\n"
    "3. Collinearità e Pruning: Rimuovi le variabili superflue o che apportano solo rumore ridondante.\n"
    "4. Niente Math-Bruteforcing: Rifiutati categoricamente di applicare trasformazioni matematiche brute (log, exp, standard scaling isolati dal contesto) sperando di trovare segnale a caso."
)

MEMORY_SYSTEM_PROMPT = (
    "Sei l'agente di memoria del sistema AutoML. Il tuo compito è analizzare la cronologia delle iterazioni precedenti "
    "per estrarre pattern ricorrenti e informazioni utili per le iterazioni future.\n\n"
    "COMPITI:\n"
    "1. Conserva traccia di tutte le iterazioni, metriche, feature utilizzate e modelli scelti.\n"
    "2. Estrai pattern: quali feature hanno avuto alta importanza in più iterazioni? Quali modelli hanno performato meglio?\n"
    "3. Fornisci contesto contestuale per le nuove iterazioni basandoti sulla memoria storica.\n\n"
    "PRINCIPI:\n"
    "- Sii conciso ma informativo.\n"
    "- Dai priorità ai pattern che si ripetono.\n"
    "- Non inventare informazioni non presenti nella cronologia."
)

MODEL_SELECTOR_SYSTEM_PROMPT = (
    "Sei l'agente di selezione modello del sistema AutoML. Il tuo compito è analizzare i dati e il contesto "
    "per raccomandare il modello di machine learning più adatto.\n\n"
    "COMPITI:\n"
    "1. Analizza la distribuzione dei dati, il tipo di target e le caratteristiche del dataset.\n"
    "2. Considera il contesto di business e i requisiti di interpretabilità.\n"
    "3. Raccomanda un modello specifico con motivazione dettagliata.\n\n"
    "MODELLI DISPONIBILI:\n"
    "- Regressione Logistica: problema lineare, alta interpretabilità, baseline.\n"
    "- Random Forest: robustezza, gestione non-linearità, media interpretabilità.\n"
    "- Gradient Boosting (XGBoost/LightGBM): alta performance, overfitting risk, bassa interpretabilità.\n\n"
    "PRINCIPI:\n"
    "- Justifica ogni raccomandazione con dati e ragionamento.\n"
    "- Considera il trade-off performance/interpretabilità.\n"
    "- Rispondi in formato JSON strutturato."
)

PRUNING_SYSTEM_PROMPT = (
    "Sei l'agente di pruning delle feature del sistema AutoML. Il tuo compito è analizzare l'importanza delle feature "
    "e identificare quelle da rimuovere per migliorare le performance.\n\n"
    "COMPITI:\n"
    "1. Analizza feature_importance e correlazioni.\n"
    "2. Identifica feature rumorose, ridondanti o con bassa importanza.\n"
    "3. Genera una lista di feature da rimuovere con motivazione.\n\n"
    "CRITERI DI PRUNING:\n"
    "- Bassa importanza (< 0.01)\n"
    "- Alta correlazione con altre feature (> 0.9)\n"
    "- Feature con molti missing value (> 50%)\n"
    "- Feature con single categoria dominante (> 95%)\n\n"
    "PRINCIPI:\n"
    "- Sii conservativo: meglio rimuovere meno feature che troppe.\n"
    "- Justifica ogni decisione di rimozione.\n"
    "- Rispondi in formato JSON con lista feature da rimuovere."
)

STRATEGY_SYSTEM_PROMPT = (
    "Sei l'agente di strategia del sistema AutoML. Il tuo compito è generare una strategia di business "
    "per la creazione di feature derivate basata sul dominio del problema.\n\n"
    "COMPITI:\n"
    "1. Analizza il glossario semantico e lo schema dei dati.\n"
    "2. Identifica fenomeni di business rilevanti per la predizione.\n"
    "3. Proponi strategie di feature crossing, ratio e aggregazioni semantiche.\n\n"
    "PRINCIPI:\n"
    "- Focus sulla semantica, non sulla matematica.\n"
    "- Proponi feature che riflettono dinamiche del mondo reale.\n"
    "- Evita trasformazioni elementari (log, exp, polinomi)."
)

CODE_SYSTEM_PROMPT = (
    "Sei l'agente di generazione codice del sistema AutoML. Il tuo compito è generare codice Python "
    "per il feature engineering e il training del modello.\n\n"
    "COMPITI:\n"
    "1. Implementa le feature derivate dalla strategia di business.\n"
    "2. Applica il pruning delle feature identificate.\n"
    "3. Gestisci missing value e codifica variabili categoriche.\n"
    "4. Implementa il modello di machine learning scelto.\n\n"
    "PRINCIPI:\n"
    "- Scrivi codice pulito, modulare e difensivo.\n"
    "- Gestisci sempre null, nan e infinite.\n"
    "- Mantieni la colonna target intatta durante le trasformazioni.\n"
    "- NON usare trasformazioni matematiche elementari (log, exp, polinomi)."
)

EVALUATOR_SYSTEM_PROMPT = (
    "Sei l'agente di valutazione del sistema AutoML. Il tuo compito è analizzare i risultati dell'iterazione "
    "e generare riflessioni per migliorare le performance nelle iterazioni successive.\n\n"
    "COMPITI:\n"
    "1. Analizza il report di valutazione (metriche, feature importance).\n"
    "2. Interpreta i grafici delle distribuzioni delle feature.\n"
    "3. Identifica pattern di successo e fallimento.\n"
    "4. Proponi azioni concrete per migliorare.\n\n"
    "PRINCIPI:\n"
    "- Sii analitico e tecnico.\n"
    "- Collega le osservazioni ai risultati concreti.\n"
    "- Proponi azioni specifiche e actionable.\n"
    "- Non proporre mai trasformazioni matematiche elementari."
)

def get_business_strategy_prompt(glossary: str, data_schema: str, data_sample: str) -> str:
    return f"""
        Analizza i seguenti metadati di progetto:
        
        # GLOSSARIO SEMANTICO
        {glossary}
        
        # SCHEMA E SAMPLE DATI
        {data_schema}
        {data_sample}
        
        Obiettivo: Devi ragionare sul contesto di business e sulla tipologia dei dati a disposizione per generare feature predittive dal punto di vista del dominio. Ragiona su quali fenomeni di business possono essere predittivi della variabile target e indica come trasformare i dati di conseguenza. Considera le best practice per la creazione di un modello di machine learning dato questo specifico contesto e il tipo di target.
        
        Restituisci ESCLUSIVAMENTE un JSON valido (senza blocchi markdown) con una chiave:
        - "business_strategy": Stringa con 3-5 strategie di business applicabili tramite feature crossing, ratio, o aggregazioni che mappano dinamiche del mondo reale aderenti allo specifico dominio in analisi (es. indici di rischio, propensione all'acquisto, efficienza operativa, ecc.). TASSATIVO: Non proporre semplici trasformate (es. logaritmo, polinomi) di feature esistenti.\n
        - "model selection": Stringa con 2 o 3 modelli di machine learning adatti al problema, con una breve motivazione per ciascuno.
"""

def get_reflection_prompt(iter_num: int, evaluation_report: str, glossary: str, feature_importance: str | None = None, trend_context: str = "", successful_patterns: str = "", failed_patterns: str = "") -> str:
    fi_section = f"""
        # FEATURE IMPORTANCE (dal modello)
        {feature_importance}
        
        Analizza l'importanza delle feature per capire quali variabili il modello considera più utili per la predizione.
        Confronta l'importanza del modello con le correlazioni: una feature con alta correlazione ma bassa importanza potrebbe essere ridondante.
    """ if feature_importance else ""
    
    trend_section = f"""
        # TENDENZA METRICA (storico iterazioni precedenti)
        {trend_context}
        
        Analizza la traiettoria: il modello sta migliorando, stagnando, o peggiorando?
    """ if trend_context else ""
    
    patterns_section = f"""
        # FEATURE DI SUCCESSO (storico)
        {successful_patterns}
        
        # FEATURE FALLITE (storico)
        {failed_patterns}
    """ if (successful_patterns or failed_patterns) else ""
    
    return f"""
        Analizza i risultati dell'Iterazione {iter_num}:
        
        # REPORT DI VALUTAZIONE
        {evaluation_report}
        
        # GLOSSARIO
        {glossary}
        
        # IMMAGINI ALLEGATE
        Sono allegate immagini che mostrano le distribuzioni delle feature (fino a 10):
        
        VIOLIN PLOTS (feature numeriche):
        - Ogni violino rappresenta la distribuzione per classe target (0/1)
        - Asse X: classe del target
        - Asse Y: valori normalizzati della feature
        - Cerca: SEPARAZIONE NETTA tra i due violini = alto potere discriminante
        - Attenzione a: SOVRAPPOSIZIONE = feature rumorosa, OUTLIER = possibile data leakage
        
        BAR CHARTS (feature categoriche):
        - Mostra il tasso di default per ogni categoria
        - Cerca: GRADIENTE MONOTONO (crescente/decrescente) = pattern di business chiaro
        - Cerca: DIFFERENZE SIGNIFICATIVE tra categorie = discriminazione forte
        - Attenzione a: categorie con POCHI CAMPIONI (n<30) = stime instabili
        
        COMBINED ANALYSIS:
        1. Confronta i grafici con la feature_importance: le feature visivamente buone sono anche importanti per il modello?
        2. I pattern che vedi nei grafici confermano o smentiscono le correlazioni calcolate?
        3. Gli outlier visibili nei grafici spiegano anomalie nelle metriche del modello?
        
        {fi_section}
        {trend_section}
        {patterns_section}
        
        # TASK - PARTE 1: ANALISI TREND
        Analizza la tendenza del modello basandoti sullo storico delle metriche:
        1. Il modello sta migliorando, stagnando (plateau), o declinando?
        2. Se stagnante o declinante, cosa è cambiato rispetto alle iterazioni precedenti?
        3. Quali decisioni delle passate iterazioni potrebbero aver causato la stagnazione?
        
        # TASK - PARTE 2: RIFLESSIONE PROFonda
        Rifletti in maniera approfondita sulla run precedente, con particolare attenzione alla costruzione di feature derivate e alla scelta del modello di machine learning più opportuno:
        1. Quali feature derivate hanno dimostrato alta importanza prestazionale e qual è la loro logica fenomenologica di dominio? Cosa mostrano i grafici in termini di separabilità tra classi target?
        2. Quali variabili (vecchie o derivate) hanno importanza nulla, o causano solo ridondanza/collinearità, e vanno eliminate nel pruning?
        
        # TASK - PARTE 3: RAGIONAMENTO CONTROFATTUALE
        1. Quali combinazioni alternative di feature avrebbero potuto essere provate? Quali interazioni non ancora esplorate potrebbero essere predittive?
        2. Quale logica di business non ancora esplorata potrebbe aiutare?
        3. Cosa sarebbe successo se avessi rimosso le feature a bassa importanza invece di aggiungerne di nuove?
        
        # TASK - PARTE 4: ABLATION ANALYSIS
        1. Quali feature potresti rimuovere mantenendo la maggior parte del potere predittivo?
        2. Qual è il contributo marginale delle top feature rispetto alle altre?
        
        # TASK - PARTE 5: PROSSIMA ITERAZIONE
        1. Basandoti sulle best practice di ML per il contesto corrente e sulle distribuzioni visibili nei grafici, quali nuove logiche di business o incroci andrebbero creati nella prossima run per massimizzare il segnale?
        2. Se la tendenza è stagnante, potresti provare un approccio radicalmente diverso?
        3. Seleziona un unico modello, lineare o non lineare, sulla base dei risultati del report di valutazione. Scegli tra regressione logistica, eventualmente con penalizzazione L1/L2, Random Forest o BoostedTree.

        ATTENZIONE: Non proporre MAI di costruire trasformate elementari (logaritmi, exp, polinomi, ecc.) di feature esistenti. Il focus deve essere al 100% sulla semantica dei dati.
        ATTENZIONE: Non testare più modelli
        
        Rispondi in modo tecnico e analitico (max 800 parole). NON scrivere codice Python.
    """

def get_code_generation_prompt(business_strategy: str, reflection_text: str, last_code: str, last_error: str = None) -> str:
    prompt = f"""
        Genera il nuovo file `dynamic_features.py`.
        
        # BUSINESS FEATURE STRATEGY (Ancoraggio Top-Down)
        {business_strategy}
        
        # RIFLESSIONE SUL RUN PRECEDENTE (Ancoraggio Bottom-Up)
        {reflection_text}
        
        # CODICE PRECEDENTE
        {last_code}
    """
    
    if last_error:
        prompt += f"\nATTENZIONE - CRITICAL FIX REQURED: L'esecuzione precedente ha generato questo errore:\n{last_error}\nCorreggi il codice per gestire questo crash (es. divisioni per zero, nan, tipi errati).\n"
        
    prompt += """
        REGOLE FONDAMENTALI:
        1. Esegui pruning delle feature irrilevanti individuate nella riflessione.
        2. Gestisci sempre i missing value in ottica di best practice (es. mediana per cont, 'Unknown' per cat, o imputation condizionata).
        3. Codifica le variabili categoriche IN MODO INTELLIGENTE in base al modello:
           - Per modelli tree-based (RandomForest, HistGradientBoostingClassifier, GradientBoosting, XGBoost, LightGBM, ExtraTrees): NON usare OneHotEncoder! Usa OrdinalEncoder (codifica intera) oppure per HistGradientBoostingClassifier usa il parametro categorical_features e passa le categoriche direttamente senza alcuna codifica.
           - Per modelli lineari (LogisticRegression, LinearRegression, Ridge, Lasso): usa OneHotEncoder.
           - NON usare mai LabelEncoder (non supporta unknown values).
        4. COSTRUISCI FEATURE DERIVATE CHE RIFLETTONO I FENOMENI DI BUSINESS discusso nella riflessione.
        5. TASSATIVO: NON COSTRUIRE MAI trasformate elementari di singole feature esistenti (assolutamente NO a `np.log`, polinomi al quadrato o radici quadrate). Tutte le tue feature aggiunte devono fondere più colonne o applicare vere logiche operative.
        6. Scegli il modello di machine learning più adatto al problema.
        7. Il codice deve contenere `def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:` e `def get_model():`.
        8. CRITICO: All'inizio di apply_feature_engineering(), PRESERVA la colonna target (default_flag) prima di qualsiasi trasformazione e RIPRISTINALA ALLA FINE prima di restituire il DataFrame. Usa questo pattern:
           ```python
           # Preserva la colonna target
           target_col = None
           for t in TARGET_CANDIDATES:
               if t in df.columns:
                   target_col = df[t].copy()  # COPIA i valori, non il nome!
                   break
           # ... tua logica di feature engineering ...
           # Riproduci la colonna target
           if target_col is not None:
               df["default_flag"] = target_col
           return df
           ```
        
        Restituisci SOLO codice Python raw, formattato correttamente. NON includere testo discorsivo o blocchi markdown (```python).
    """
    return prompt


def _extract_relevant_function(error_message: str, code: str) -> str:
    import re
    match = re.search(r'in (\w+)\s*$|File ".+?", line \d+, in (\w+)', error_message, re.MULTILINE)
    func_name = match.group(1) or match.group(2) if match else None
    
    if func_name and func_name in ['apply_feature_engineering', 'get_model']:
        lines = code.split('\n')
        func_start = -1
        func_lines = []
        in_function = False
        indent_level = None
        
        for i, line in enumerate(lines):
            if f'def {func_name}' in line:
                func_start = i
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                func_lines.append(line)
                continue
            
            if in_function:
                if line.strip() and not line.startswith(' ' * indent_level) and not line.startswith('\t'):
                    break
                func_lines.append(line)
        
        if func_start >= 0:
            return '\n'.join(func_lines)
    
    return code


def get_error_fix_prompt(error_message: str, previous_code: str) -> str:
    relevant_code = _extract_relevant_function(error_message, previous_code)
    
    return f"""Fix ONLY the error in this code. DO NOT change anything else.

ERROR:
{error_message}

CODE TO FIX:
{relevant_code}

Rules:
1. Make minimal changes to fix the error
2. Do NOT add new features or change strategy
3. Preserve all existing feature engineering logic
4. Only fix what's broken

Return ONLY the fixed Python code."""


def get_model_selection_prompt(
    data_schema: str,
    data_sample: str,
    glossary: str,
    memory_context: str | None = None,
    feature_importance: dict | None = None
) -> str:
    fi_section = ""
    if feature_importance:
        fi_str = "\n".join(f"  - {k}: {v:.4f}" for k, v in list(feature_importance.items())[:10])
        fi_section = f"\n# FEATURE IMPORTANCE (ultimo modello)\n{fi_str}\n"

    mem_section = f"\n# MEMORIA STORICA\n{memory_context}\n" if memory_context else ""

    return f"""
        Analizza il dataset e raccomanda il modello di machine learning più adatto.

        # SCHEMA DATI
        {data_schema}

        # SAMPLE DATI
        {data_sample}

        # GLOSSARIO
        {glossary}
        {mem_section}
        {fi_section}

        TASK:
        1. Analizza le caratteristiche del dataset (numero feature, tipo target, distribuzione).
        2. Considera i pattern dalla memoria storica (se disponibili).
        3. Considera l'interpretabilità richiesta dal business.
        4. Raccomanda un singolo modello con motivazione dettagliata.

        Rispondi ESCLUSIVAMENTE con JSON valido (no markdown):
        {{
            "recommended_model": "nome del modello",
            "rationale": "motivazione dettagliata (2-3 frasi)",
            "backup_model": "eventuale modello alternativo"
        }}
    """


def get_pruning_prompt(
    feature_importance: dict,
    correlations: dict | None = None,
    memory_context: str | None = None
) -> str:
    corr_section = ""
    if correlations:
        corr_str = "\n".join(f"  - {k} <-> {v:.3f}" for k, v in list(correlations.items())[:15])
        corr_section = f"\n# CORRELAZIONI TRA FEATURE\n{corr_str}\n"

    mem_section = f"\n# MEMORIA STORICA\n{memory_context}\n" if memory_context else ""

    fi_str = "\n".join(f"  - {k}: {v:.4f}" for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return f"""
        Analizza le feature e identifica quelle da rimuovere.

        # FEATURE IMPORTANCE
        {fi_str}
        {corr_section}
        {mem_section}

        CRITERI DI PRUNING:
        - Rimuovi feature con importanza < 0.01
        - Rimuovi feature con correlazione > 0.9 (ridondanza)
        - Rimuovi feature con singola categoria dominante (> 95%)
        - Rimuovi feature identificate come problematiche nella memoria storica

        TASK:
        1. Identifica le feature da rimuovere.
        2. Per ogni feature, indica il motivo della rimozione.
        3. Suggerisci eventuali azioni correttive.

        Rispondi ESCLUSIVAMENTE con JSON valido (no markdown):
        {{
            "features_to_drop": ["feature1", "feature2"],
            "rationale": {{
                "feature1": "motivo rimozione",
                "feature2": "motivo rimozione"
            }},
            "suggestions": ["eventuali suggerimenti aggiuntivi"]
        }}
    """


def get_iterative_strategy_prompt(
    glossary: str,
    data_schema: str,
    data_sample: str,
    memory_context: str,
    last_iteration_results: dict | None = None,
    trend_context: str = "",
    strategy_context: str = ""
) -> str:
    last_results_section = ""
    if last_iteration_results:
        metric = last_iteration_results.get('metric', 'N/A')
        features = ", ".join(last_iteration_results.get('features_used', [])[:5])
        model = last_iteration_results.get('model_used', 'N/A')
        last_results_section = f"\n# ULTIMA ITERAZIONE\n- Metrica: {metric}\n- Features usate: {features}\n- Modello: {model}\n"

    trend_section = f"\n# TENDENZA RECENTE\n{trend_context}\n" if trend_context else ""
    strategy_sec = f"\n# MIGLIOR STRATEGIA PRECEDENTE\n{strategy_context}\n" if strategy_context else ""

    return f"""
        Analizza il contesto e genera/aggiorna la strategia di business.

        # GLOSSARIO SEMANTICO
        {glossary}

        # SCHEMA E SAMPLE DATI
        {data_schema}
        {data_sample}

        # MEMORIA STORICA
        {memory_context}
        {last_results_section}
        {trend_section}
        {strategy_sec}

        TASK:
        1. Considera cosa ha funzionato nelle iterazioni precedenti.
        2. Analizza la tendenza attuale: il modello sta migliorando, stagnando o peggiorando?
        3. Se la tendenza è stagnante/declinante, considera un approccio radicalmente diverso.
        4. Identifica nuove opportunità basate sui pattern memorizzati e sulla tendenza.
        5. Aggiorna o conferma la strategia di business.
        6. Proponi 2-3 nuove feature che sfruttino i pattern identificati.

        Restituisci ESCLUSIVAMENTE JSON valido (no markdown):
        {{
            "business_strategy": "strategia aggiornata",
            "new_feature_ideas": ["idea1", "idea2", "idea3"],
            "model_selection": "modello consigliato"
        }}
    """
