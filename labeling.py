# data labeling using Gemini AI model
from google import genai
from google.genai import types
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()
gem_api = os.getenv('GEMINI_API')
df = pd.read_csv('datasets/comments.csv')
comments_list = df['text'].tolist()
client = genai.Client(api_key=gem_api)
# Function for classification into JSON format (Prompt is in Indonesian [ID])
def label_batch_comments(batch_texts):
    prompt = f"""
    Kamu adalah asisten pengolah data. Tugasmu adalah melakukan klasifikasi sentimen/sikap dari komentar netizen terhadap tantangan "Uninstall TikTok & max medsos 15 menit/hari".

    Aturan Labeling:
    - Label 1 (PRO): Setuju, mau mencoba, merasa realistis, mendukung, atau sudah pernah menerapkan tantangan tersebut.
    - Label 0 (KONTRA/OOT): Tidak setuju, merasa mustahil/tidak realistis, mengeluh, skeptis, menolak, atau membahas hal lain di luar konteks tantangan.

    Daftar Komentar:
    {json.dumps(batch_texts, ensure_ascii=False, indent=2)}

    Kembalikan respons HANYA dalam format JSON list berisi angka 0 atau 1 sesuai urutan komentar input. Contoh output: [1, 0, 0, 1, 1]
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", 
                temperature=0.0 
            )
        )
        labels = json.loads(response.text)
        return labels
    except Exception as e:
        print(f"Error when labeling: {e}")
        return [0] * len(batch_texts)

BATCH_SIZE = 20
all_labels = []

print(f"Start labelling process for {len(comments_list)} comments...")

for i in range(0, len(comments_list), BATCH_SIZE):
    batch = comments_list[i:i + BATCH_SIZE]
    print(f"Processing lines {i} until {i + len(batch)}...")
    
    labels = label_batch_comments(batch)
    if len(labels) == len(batch):
        all_labels.extend(labels)
    else:
        print(f"Warning: Output size does not match at batch index {i}. Doing fallback.")
        all_labels.extend([0] * len(batch))
        
    time.sleep(1) 
    
df['label'] = all_labels
df.to_csv('datasets/comments_labeled.csv', index=False, encoding='utf-8-sig')

print("Done!")