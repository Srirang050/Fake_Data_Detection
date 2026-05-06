import pandas as pd, os, sys

AI_PATH  = "data/dataset1ai.csv"
REAL_PATH= "data/dataset2real.csv"
OUT_PATH = "data/combined_texts.csv"

# helper to try reading with different encodings
def read_csv_fallback(path):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_exc = e
    raise last_exc

def pick_text_col(df):
    # common names to search for (ordered)
    candidates = ["text", "content", "review", "sentence", "body", "message", "generated"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: prefer the first string-like column
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            return c
    raise ValueError("Could not find a text column. Available columns: " + ", ".join(df.columns))

def main():
    if not os.path.exists(AI_PATH) or not os.path.exists(REAL_PATH):
        print("ERROR: Ensure files exist:", AI_PATH, "and", REAL_PATH)
        sys.exit(2)

    print("Reading", AI_PATH)
    a = read_csv_fallback(AI_PATH)
    print("Reading", REAL_PATH)
    b = read_csv_fallback(REAL_PATH)

    # detect text columns
    a_text_col = pick_text_col(a)
    b_text_col = pick_text_col(b)
    print("Detected text columns -> AI:", a_text_col, "Real:", b_text_col)

    # rename to 'text'
    a = a.rename(columns={a_text_col: "text"})
    b = b.rename(columns={b_text_col: "text"})

    # add/normalize label
    a["label"] = 1
    b["label"] = 0

    # keep only text+label
    a = a[["text","label"]]
    b = b[["text","label"]]

    # concat + shuffle
    df = pd.concat([a,b], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("Saved combined dataset to", OUT_PATH)
    print("Sizes -> AI:", len(a), "Real:", len(b), "Total:", len(df))
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
