import pandas as pd
import os

# ── Configuration ─────────────────────────────────────────────────────────────
# Change these paths to match your machine
CSV_INPUT  = r"C:\Users\space\Desktop\Baljka_diplom\test\test.csv"
START_FROM = 8000   # new starting number  (img_008000, img_008001, ...)

# Output saved next to the input file
CSV_OUTPUT = CSV_INPUT.replace(".csv", f"_from_{START_FROM:06d}.csv")

# ── Load ──────────────────────────────────────────────────────────────────────
if not os.path.isfile(CSV_INPUT):
    print(f"ERROR: File not found:\n  {CSV_INPUT}")
    exit(1)

df = pd.read_csv(CSV_INPUT)
print(f"Loaded  : {CSV_INPUT}")
print(f"Rows    : {len(df)}")
print(f"Columns : {list(df.columns)}")
print()

# ── Preview before renaming ───────────────────────────────────────────────────
print("Before renaming:")
print(f"  First : {df['image_name'].iloc[0]}")
print(f"  Last  : {df['image_name'].iloc[-1]}")

# ── Rename image_name column ──────────────────────────────────────────────────
# Extracts the 6-digit number from img_NNNNNN.png, adds START_FROM, reformats.
def renumber(name: str) -> str:
    num = int(name.replace("img_", "").replace(".png", ""))
    return f"img_{num + START_FROM:06d}.png"

df["image_name"] = df["image_name"].apply(renumber)

# ── Preview after renaming ────────────────────────────────────────────────────
print()
print("After renaming:")
print(f"  First : {df['image_name'].iloc[0]}")
print(f"  Last  : {df['image_name'].iloc[-1]}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(CSV_OUTPUT, index=False)
print()
print(f"Saved   : {CSV_OUTPUT}")
print(f"Done — {len(df)} rows renumbered starting from img_{START_FROM:06d}.png")
