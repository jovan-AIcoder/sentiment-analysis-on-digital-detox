import os
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load Environment Variables (.env)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY NOT FOUND")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.8-flash")

# 2. Input & Output
INPUT_FILE = "datasets/comments_labeled.csv"
OUTPUT_FILE = "datasets/comments_labeled_paraphrased.csv"

# 3. Paraphrasing Function (Prompt is in Indonesian)
def paraphrase_text(original_text):
    prompt = f"""
    Ubah susunan kata (paraphrase) dari teks komentar berikut ke dalam bahasa gaul/santai netizen Indonesia.
    
    Aturan Ketat:
    1. Hapus semua mention username (@username), nama orang, atau entitas pribadi (PII).
    2. Jaga agar MAKNA UTAMA dan TONE SENTIMEN (Pro/Kontra) tetap 100% sama dengan aslinya.
    3. Buat susunan kata dan struktur kalimatnya berbeda agar tidak bisa dicari secara eksak (exact match) di mesin pencari.
    4. Berikan HANYA hasil teks parafrasanya saja tanpa tanda petik ekstra, penjelas, atau intro.

    Teks Asli: "{original_text}"
    """
    try:
        response = model.generate_content(prompt)
        paraphrased = response.text.strip()
        if paraphrased.startswith('"') and paraphrased.endswith('"'):
            paraphrased = paraphrased[1:-1]
        return paraphrased
    except Exception as e:
        print(f"Error when paraphrasing: '{original_text}': {e}")
        return original_text # Fallback 

# 4. Execute Paraphrasing Process
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"'{INPUT_FILE}' not found.")
        return

    print(f"Reading datasets from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)

    if 'text' not in df.columns:
        raise ValueError("text column is not available.")

    print(f"Starting Paraphrasing Process for {len(df)} comments...\n")

    paraphrased_texts = []
    for idx, row in df.iterrows():
        original = row['text']
        print(f"[{idx+1}/{len(df)}] Processing...")
        print(f"  - Original      : {original}")
        
        # Panggil Gemini API
        paraphrased = paraphrase_text(original)
        paraphrased_texts.append(paraphrased)
        
        print(f"  - Paraphrased : {paraphrased}\n")
        
        time.sleep(1)

    df['text'] = paraphrased_texts

    # Save to a new CSV file
    os.makedirs('datasets', exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    main()