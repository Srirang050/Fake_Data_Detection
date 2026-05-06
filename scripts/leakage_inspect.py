# scripts/leakage_inspect.py
import pandas as pd, re
df = pd.read_csv("data/combined_texts.csv", dtype={'text':str,'label':int})
print("Total rows:", len(df), "Unique texts:", df['text'].nunique())
print("\nLabel counts:\n", df['label'].value_counts())

# top prefixes
print("\nTop 30 first-20-char prefixes:")
print(df['text'].astype(str).str[:20].value_counts().head(30).to_string())

# any exact label token present in text?
for label_token in ["0", "1", "label", "synth", "generated", "ai", "AI", "generated_by", "human", "source"]:
    mask = df['text'].astype(str).str.contains(label_token, na=False)
    if mask.any():
        print(f"\nFound token '{label_token}' in {mask.sum()} rows (sample):")
        print(df.loc[mask, 'text'].astype(str).head(5).to_list())

# regex look for obvious markers (html tags, URLs, meta tokens)
html_mask = df['text'].astype(str).str.contains(r"<\/?\w+>", regex=True, na=False)
url_mask = df['text'].astype(str).str.contains(r"https?://", regex=True, na=False)
meta_mask = df['text'].astype(str).str.contains(r"generated_by|generator|source:|SYNTH|__SYNTH__", flags=0, na=False)

print(f"\nHTML tags present in {html_mask.sum()} rows, URLs in {url_mask.sum()}, 'generated' kind markers in {meta_mask.sum()}.")
