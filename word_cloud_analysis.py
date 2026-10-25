# make a wordcloud
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Load the labeled CSV dataset
CSV_FILENAME = 'datasets/comments_labeled.csv'

try:
    df = pd.read_csv(CSV_FILENAME)
    print(f"Successfully loaded '{CSV_FILENAME}'. Total records: {len(df)}")
except FileNotFoundError:
    print(f"Error: File '{CSV_FILENAME}' not found. Please verify the path.")
    exit()

# 2. Basic Indonesian Stopwords List (Common filler words)
INDONESIAN_STOPWORDS = set([
    'yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu', 'dengan', 'untuk', 'pada',
    'adalah', 'sebagai', 'juga', 'akan', 'bisa', 'ada', 'tidak', 'gak', 'nggak',
    'ga', 'ya', 'aja', 'saja', 'sudah', 'udah', 'kalo', 'kalau', 'buat', 'sama',
    'bikin', 'jadi', 'lagi', 'deh', 'dong', 'sih', 'gue', 'gw', 'aku', 'saya',
    'lu', 'elo', 'kamu', 'kalian', 'orang', 'yg', 'dgn', 'sdh', 'tdk'
])

# 3. Text Preprocessing Function
def clean_text_for_wordcloud(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, and special characters/punctuation
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove stopwords
    words = text.split()
    filtered_words = [w for w in words if w not in INDONESIAN_STOPWORDS and len(w) > 2]
    
    return " ".join(filtered_words)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(clean_text_for_wordcloud)

# 4. Separate text by sentiment class
pro_text = " ".join(df[df['label'] == 1]['cleaned_text'])
con_text = " ".join(df[df['label'] == 0]['cleaned_text'])

# 5. Generate Word Cloud Images
def create_wordcloud(text_data, title, colormap):
    if not text_data.strip():
        print(f"Warning: No text available to generate Word Cloud for '{title}'.")
        return None
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap=colormap,
        collocations=False
    ).generate(text_data)
    return wc

wc_pro = create_wordcloud(pro_text, "Pro Group", "Greens")
wc_con = create_wordcloud(con_text, "Con/OOT Group", "Reds")

# 6. Plotting side-by-side using Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot Pro WordCloud
if wc_pro:
    axes[0].imshow(wc_pro, interpolation='bilinear')
    axes[0].set_title("Dominant Words: Pro (Class 1)", fontsize=14, weight='bold', pad=15)
axes[0].axis('off')

# Plot Con/OOT WordCloud
if wc_con:
    axes[1].imshow(wc_con, interpolation='bilinear')
    axes[1].set_title("Dominant Words: Con / Skeptical / OOT (Class 0)", fontsize=14, weight='bold', pad=15)
axes[1].axis('off')

plt.suptitle("Word Cloud Distribution Analysis by Sentiment Class", fontsize=16, weight='bold', y=0.98)
plt.tight_layout()

# Save the visualization plot
OUTPUT_IMAGE = 'figures/wordcloud_analysis.png'
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
print(f"Word Cloud plot saved successfully to '{OUTPUT_IMAGE}'.")

plt.show()