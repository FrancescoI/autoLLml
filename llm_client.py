import os
import base64
from dotenv import load_dotenv

load_dotenv()

# Esempio usando OpenAI, ma facilmente sostituibile con la libreria desiderata
from openai import OpenAI

# L'API key verrà letta sia dal .env che dalle variabili d'ambiente di sistema
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

from prompts import SYSTEM_PROMPT

def generate_response(prompt: str, system_prompt: str = None) -> str:
    """Funzione centralizzata per le chiamate all'LLM."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] Errore API LLM: {e}")
        return ""


def generate_response_with_images(prompt: str, image_paths: list, system_prompt: str = None) -> str:
    """
    Chiamata all'LLM con supporto Vision (multi-modal).
    Accetta una lista di path a immagini PNG/JPG, le codifica in base64 e le invia
    come blocchi image_url nella stessa richiesta del testo.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    # Costruiamo il contenuto multi-modale del messaggio utente
    content = [{"type": "text", "text": prompt}]
    for path in image_paths:
        try:
            with open(path, "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode("utf-8")
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            mime = f"image/{ext if ext in ('png', 'jpg', 'jpeg', 'gif', 'webp') else 'png'}"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}
            })
        except Exception as e:
            print(f"[!] Impossibile caricare immagine {path}: {e}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] Errore API LLM (vision): {e}")
        # Fallback: ritenta senza immagini
        return generate_response(prompt, system_prompt)