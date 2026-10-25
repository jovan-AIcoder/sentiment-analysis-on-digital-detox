# plot the distribution using pie chart
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the labeled CSV file
CSV_FILENAME = 'datasets/comments_labeled.csv'

try:
    df = pd.read_csv(CSV_FILENAME)
    print(f"Successfully loaded '{CSV_FILENAME}'. Total records: {len(df)}")
except FileNotFoundError:
    print(f"Error: File '{CSV_FILENAME}' not found. Please run the labeling script first.")
    exit()

# 2. Calculate counts and probabilities for class 1 (Pro) and class 0 (Con/OOT)
# Label 1: Pro (Digital Detox Challenge)
# Label 0: Con / Skeptical / Off-Topic
value_counts = df['label'].value_counts()

count_pro = value_counts.get(1, 0)
count_con = value_counts.get(0, 0)
total_comments = len(df)

if total_comments == 0:
    print("Error: The CSV file is empty.")
    exit()

p_pro = count_pro / total_comments
p_con = count_con / total_comments

# 3. Compute Shannon Entropy (H) in bits (base 2)
# H(X) = - sum(P(x) * log2(P(x)))
entropy = 0.0
probabilities = [p_pro, p_con]

for p in probabilities:
    if p > 0:  # Avoid log2(0)
        entropy -= p * np.log2(p)

# Print theoretical summary to console
print("\n" + "="*40)
print("       DISTRIBUTION ANALYSIS SUMMARY       ")
print("="*40)
print(f"Total Comments Analyzed : {total_comments}")
print(f"Pro (Class 1) Count     : {count_pro} ({p_pro * 100:.2f}%)")
print(f"Con/OOT (Class 0) Count : {count_con} ({p_con * 100:.2f}%)")
print(f"Shannon Entropy (H)     : {entropy:.4f} bits (Max: 1.0000 bit)")
print("="*40 + "\n")

# 4. Plotting Pie Chart using Matplotlib
labels = ['Pro (Accept Challenge)', 'Con / Skeptical / OOT']
sizes = [count_pro, count_con]
colors = ['#2ecc71', '#e74c3c']  # Green for Pro, Red for Con
explode = (0.05, 0)  # Slightly separate the 'Pro' slice for emphasis

plt.figure(figsize=(8, 7))

wedges, texts, autotexts = plt.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    textprops=dict(color="black", fontsize=11),
    shadow=True
)

# Format percentage text styling inside slices
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')
    autotext.set_fontsize(12)

# Title and Entropy annotation block on the chart
plt.title("Netizen Sentiment Distribution\non Digital Detox Challenge", fontsize=14, pad=20, weight='bold')

# Display Entropy information in a styled text box
text_str = f"Total Sample (N): {total_comments}\nShannon Entropy (H): {entropy:.4f} bits"
plt.gcf().text(
    0.5, 0.03, text_str,
    fontsize=11,
    ha='center',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='gray', alpha=0.9)
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Leave space for the bottom text box
plt.savefig('figures/pie_chart.png', dpi=300)
# Show the chart
plt.show()