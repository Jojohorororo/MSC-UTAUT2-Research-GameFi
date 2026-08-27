

import pandas as pd

# Load  Excel file
df = pd.read_excel('utaut2_cleaned_data.xlsx')

print("=" * 70)
print("EXCEL FILE INSPECTION")
print("=" * 70)

print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "=" * 70)
print("COLUMN NAMES IN YOUR FILE:")
print("=" * 70)

for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

print("\n" + "=" * 70)
print("FIRST 5 ROWS OF DATA:")
print("=" * 70)
print(df.head())

print("\n" + "=" * 70)
print("DATA TYPES:")
print("=" * 70)
print(df.dtypes)

print("\n" + "=" * 70)
print("EXPECTED CONSTRUCT NAMES:")
print("=" * 70)
expected = ['PE', 'EE', 'SI', 'FC', 'HM', 'PV', 'HB', 'BI', 'EM', 'RP', 'TT', 'RC']
print(", ".join(expected))

print("\n" + "=" * 70)
print("MATCHING CHECK:")
print("=" * 70)

for construct in expected:
    if construct in df.columns:
        print(f"✓ {construct} - FOUND")
    else:
        # Look for partial matches
        matches = [col for col in df.columns if construct.lower() in str(col).lower()]
        if matches:
            print(f"⚠ {construct} - Not exact, but found: {matches}")
        else:
            print(f"✗ {construct} - NOT FOUND")

print("\n" + "=" * 70)