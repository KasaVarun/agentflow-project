"""One-shot: download + extract Spider databases."""
import requests, zipfile, io, os

url = "https://drive.usercontent.google.com/download?id=1TqleXec_OykOYFREKKtschzY29dUcVAQ&confirm=t"
db_root = "benchmarks/text2sql/data/spider"

print("Downloading Spider (~100MB)...")
r = requests.get(url, stream=True, timeout=300)
data = b"".join(r.iter_content(1024 * 1024))
print(f"Downloaded {len(data)//1024//1024}MB. Extracting...")

zf = zipfile.ZipFile(io.BytesIO(data))
extracted = 0
for name in zf.namelist():
    if name.endswith("/"):
        continue
    parts = name.split("/", 1)
    if len(parts) < 2 or not parts[1]:
        continue
    rel = parts[1]
    target = os.path.join(db_root, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with zf.open(name) as src, open(target, "wb") as dst:
        dst.write(src.read())
    extracted += 1
    if extracted % 200 == 0:
        print(f"  {extracted} files...")

print(f"Extracted {extracted} files.")
db_dir = os.path.join(db_root, "database")
dbs = os.listdir(db_dir) if os.path.isdir(db_dir) else []
print(f"Databases: {len(dbs)} dirs (e.g. {dbs[:3]})")
print(f"tables.json: {os.path.exists(os.path.join(db_root, 'tables.json'))}")
