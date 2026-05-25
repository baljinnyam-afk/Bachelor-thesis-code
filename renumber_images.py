import pandas as pd
import os

# ── Configuration ─────────────────────────────────────────────────────────────
# Change these two paths to match your machine
CSV_INPUT  = r"C:\Users\space\Desktop\Baljka_diplom\test\test.csv"
IMAGES_DIR = r"C:\Users\space\Desktop\Baljka_diplom\test\images"
START_FROM = 8000   # new starting number (img_008000, img_008001, ...)

# Output CSV saved next to the input file
CSV_OUTPUT = CSV_INPUT.replace(".csv", f"_from_{START_FROM:06d}.csv")


# ── Load CSV ──────────────────────────────────────────────────────────────────
if not os.path.isfile(CSV_INPUT):
    print(f"ERROR: CSV not found:\n  {CSV_INPUT}")
    exit(1)

if not os.path.isdir(IMAGES_DIR):
    print(f"ERROR: Images folder not found:\n  {IMAGES_DIR}")
    exit(1)

df = pd.read_csv(CSV_INPUT)
print(f"Loaded  : {CSV_INPUT}")
print(f"Rows    : {len(df)}")
print()

# ── Build rename map  old_name -> new_name ────────────────────────────────────
def new_name(old: str) -> str:
    num = int(old.replace("img_", "").replace(".png", ""))
    return f"img_{num + START_FROM:06d}.png"

rename_map = {row: new_name(row) for row in df["image_name"]}

# ── Preview ───────────────────────────────────────────────────────────────────
first_old = df["image_name"].iloc[0]
last_old  = df["image_name"].iloc[-1]
print(f"Before : {first_old}  ->  {last_old}")
print(f"After  : {rename_map[first_old]}  ->  {rename_map[last_old]}")
print()

# ── Check all image files exist before touching anything ─────────────────────
missing = [n for n in rename_map if not os.path.isfile(os.path.join(IMAGES_DIR, n))]
if missing:
    print(f"WARNING: {len(missing)} image files not found in {IMAGES_DIR}")
    print("  First 5 missing:")
    for m in missing[:5]:
        print(f"    {m}")
    answer = input("\nContinue anyway and skip missing files? (y/n): ").strip().lower()
    if answer != "y":
        print("Aborted.")
        exit(0)
else:
    print(f"All {len(rename_map)} image files found in images folder.")

# ── Rename image files in TWO PASSES to avoid collision ──────────────────────
# Collision example: renaming img_000001 -> img_008001 could overwrite an
# existing img_008001 if that frame already existed from a previous run.
# Pass 1: img_NNNNNN.png  -> img_NNNNNN.tmp  (stage all files)
# Pass 2: img_NNNNNN.tmp  -> img_(N+8000).png (commit to final names)

print("\nPass 1/2 - staging files to .tmp ...")
staged = []
skipped = 0
for old in rename_map:
    src = os.path.join(IMAGES_DIR, old)
    tmp = os.path.join(IMAGES_DIR, old.replace(".png", ".tmp"))
    if os.path.isfile(src):
        os.rename(src, tmp)
        staged.append((tmp, os.path.join(IMAGES_DIR, rename_map[old])))
    else:
        skipped += 1

print(f"  Staged : {len(staged)} files")
if skipped:
    print(f"  Skipped: {skipped} files not found")

print("Pass 2/2 - renaming .tmp to final names ...")
errors = 0
for tmp, final in staged:
    try:
        os.rename(tmp, final)
    except Exception as e:
        print(f"  ERROR: {os.path.basename(tmp)} -> {e}")
        errors += 1

done = len(staged) - errors
print(f"  Done   : {done} files renamed successfully")
if errors:
    print(f"  Errors : {errors}")

# ── Update and save CSV ───────────────────────────────────────────────────────
df["image_name"] = df["image_name"].apply(lambda n: rename_map.get(n, n))
df.to_csv(CSV_OUTPUT, index=False)

print()
print(f"CSV saved : {CSV_OUTPUT}")
print(f"\nSummary")
print(f"  Images renamed : {done}")
print(f"  CSV rows       : {len(df)}")
print(f"  First entry    : {df['image_name'].iloc[0]}")
print(f"  Last  entry    : {df['image_name'].iloc[-1]}")
print("\nDone.")
