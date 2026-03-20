# prompts.py

SYSTEM_PROMPT = (
    "Sei un Senior Data Scientist esperto in Machine Learning e Artificial Intelligence, con una forte mentalità orientata al Business e al Domain Knowledge.\n\n"
    "I TUOI PRINCIPI FONDAMENTALI:\n"
    "1. Semantica prima della statistica: Ogni feature che crei deve avere un senso nel mondo reale legato allo specifico dominio del problema (es. se è marketing, calcoli una frequency; se è rischio creditizio, calcoli debt-to-income). Niente logaritmi o altre trasformazioni dei dati senza un senso logico.\n"
    "2. Clean Code & robustezza: Scrivi codice Python pulito, modulare e difensivo (usando pandas e scikit-learn). Gestisci sempre null, nan e infinite in modo logico e coerente con la best-practice.\n"
    "3. Collinearità e Pruning: Rimuovi le variabili superflue o che apportano solo rumore ridondante.\n"
    "4. Niente Math-Bruteforcing: Rifiutati categoricamente di applicare trasformazioni matematiche brute (log, exp, standard scaling isolati dal contesto) sperando di trovare segnale a caso."
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

def get_reflection_prompt(iter_num: int, evaluation_report: str, glossary: str, feature_importance: str | None = None) -> str:
    fi_section = f"""
        # FEATURE IMPORTANCE (dal modello)
        {feature_importance}
        
        Analizza l'importanza delle feature per capire quali variabili il modello considera più utili per la predizione.
        Confronta l'importanza del modello con le correlazioni: una feature con alta correlazione ma bassa importanza potrebbe essere ridondante.
    """ if feature_importance else ""
    
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
        
        # TASK:
        Rifletti in maniera approfondita sulla run precedente, con particolare attenzione alla costruzione di feature derivate e alla scelta del modello di machine learning più opportuno:
        1. Quali feature derivate hanno dimostrato alta importanza prestazionale e qual è la loro logica fenomenologica di dominio? Cosa mostrano i grafici in termini di separabilità tra classi target?
        2. Quali variabili (vecchie o derivate) hanno importanza nulla, o causano solo ridondanza/collinearità, e vanno eliminate nel pruning?
        3. Basandoti sulle best practice di ML per il contesto corrente e sulle distribuzioni visibili nei grafici, quali nuove logiche di business o incroci andrebbero creati nella prossima run per massimizzare il segnale?
        4. Seleziona un unico modello, lineare o non lineare, sulla base dei risultati del report di valutazione. Scegli tra regressione logistica, eventualmente con penalizzazione L1/L2, Random Forest o BoostedTree.

        ATTENZIONE: Non proporre MAI di costruire trasformate elementari (logaritmi, exp, polinomi, ecc.) di feature esistenti. Il focus deve essere al 100% sulla semantica dei dati.
        ATTENZIONE: Non testare più modelli
        
        Rispondi in modo tecnico e analitico (max 500 parole). NON scrivere codice Python.
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
        3. Codifica le variabili categoriche adeguatamente.
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
