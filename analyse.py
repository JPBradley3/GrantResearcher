import pandas as pd

df = pd.read_csv("output/grants_fleshed.csv")
print("SHAPE:", df.shape)
print("COLUMNS:", df.columns.tolist())

print("\n=== FILL RATES ===")
for col in df.columns:
    filled = df[col].notna().sum()
    print(f"  {col}: {filled}/44 ({filled/44*100:.0f}%)")

print("\n=== AMOUNTS ===")
for _, r in df[df["llm_amount"].notna()].iterrows():
    print(f"  {r['url'].split('/')[2]}: {r['llm_amount']}")

print("\n=== DEADLINES ===")
for _, r in df[df["llm_deadline"].notna()].iterrows():
    print(f"  {r['url'].split('/')[2]}: {str(r['llm_deadline']).strip()}")

print("\n=== INVITE ONLY ===")
for _, r in df[df["llm_eligibility"].str.startswith("INVITE", na=False)].iterrows():
    print(f"  {r['url'].split('/')[2]}")

print("\n=== STILL EMPTY (no LLM data at all) ===")
empty = df[df["summary"].isna() & df["llm_amount"].isna() & df["llm_deadline"].isna() & df["llm_eligibility"].isna()]
for _, r in empty.iterrows():
    print(f"  {r['url']}")
