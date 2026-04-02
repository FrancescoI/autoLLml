import pandas as pd
from utils.config import get_paths


class DataContextProvider:
    """Gestisce il caricamento e la fornitura di glossario, schema dati e campioni."""

    def __init__(self, paths):
        self.paths = paths
        self.glossary = self._load_glossary()
        self.data_schema, self.data_sample = self._load_data()

    def _load_glossary(self) -> str:
        try:
            with open(self.paths.glossary, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"[!] Glossario non trovato in {self.paths.glossary}")
            return ""

    def _load_data(self) -> tuple[str, str]:
        try:
            df = pd.read_csv(self.paths.data, encoding="latin-1")
            data_schema = str(df.dtypes.to_dict())
            data_sample = str(df.head(1).to_dict())
            return data_schema, data_sample
        except FileNotFoundError:
            print(f"[!] Assicurati di inserire il file in {self.paths.data} prima di avviare.")
            return "Dati non caricati.", "N/A"

    def get_context(self) -> dict:
        return {
            "glossary": self.glossary,
            "data_schema": self.data_schema,
            "data_sample": self.data_sample
        }