import pandas as pd
import ast
from collections import Counter
import os

def parse_pairs(path, col_asp='aspect_terms', col_pol='at_polarity'):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []
    df = pd.read_csv(path)
    if col_asp not in df.columns and 'aspect_term' in df.columns:
        col_asp = 'aspect_term'
    pairs = []
    for idx, row in df.iterrows():
        raw_asps = row[col_asp]
        raw_pols = row[col_pol]
        text = str(row['sentence'])
        try:
            if isinstance(raw_asps, str) and raw_asps.startswith('['):
                asps = ast.literal_eval(raw_asps)
            elif isinstance(raw_asps, list):
                asps = raw_asps
            else:
                asps = []
            if isinstance(raw_pols, str) and raw_pols.startswith('['):
                pols = ast.literal_eval(raw_pols)
            elif isinstance(raw_pols, list):
                pols = raw_pols
            else:
                pols = []
        except Exception:
            continue
        for a, p in zip(asps, pols):
            a_clean = str(a).strip().strip("\"'")
            p_clean = str(p).strip().strip("\"'").lower()
            if a_clean and p_clean and p_clean != 'conflict':
                pairs.append({'sentence': text, 'aspect': a_clean, 'polarity': p_clean})
    return pairs

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_absa_mix")
rest_train = parse_pairs(os.path.join(base, "Complete_restaurant_train.csv"))
lap_train = parse_pairs(os.path.join(base, "complete_laptop_train.csv"))
rest_test = parse_pairs(os.path.join(base, "Complete_restaurant_test.csv"))
lap_test = parse_pairs(os.path.join(base, "complete_laptop_test.csv"))

all_train = rest_train + lap_train
all_test = rest_test + lap_test

print(f"Restaurant Train Pairs: {len(rest_train)}")
print(f"Laptop Train Pairs:     {len(lap_train)}")
print(f"Total Train Pairs:      {len(all_train)}")
print(f"Restaurant Test Pairs:  {len(rest_test)}")
print(f"Laptop Test Pairs:      {len(lap_test)}")
print(f"Total Test Pairs:       {len(all_test)}")

print("Train Polarity Distribution:", Counter(p['polarity'] for p in all_train))
print("Test Polarity Distribution:", Counter(p['polarity'] for p in all_test))

print("\nSample extracted pairs:")
for s in all_train[:4]:
    print(" - Text:", s['sentence'][:60] + "...")
    print("   Aspect:", s['aspect'], "-> Polarity:", s['polarity'])
