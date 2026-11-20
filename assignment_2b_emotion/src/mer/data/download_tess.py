"""Download TESS (Toronto Emotional Speech Set) from the official UofT Borealis Dataverse.

DOI: 10.5683/SP2/E8H2MF  (same 2800 stimuli as the Kaggle mirror referenced in the assignment).
Usage: python -m mer.data.download_tess --out data/raw/TESS
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://borealisdata.ca/api"
DOI = "doi:10.5683/SP2/E8H2MF"


def list_files():
    url = f"{API}/datasets/:persistentId/?persistentId={DOI}"
    with urllib.request.urlopen(url, timeout=60) as r:
        meta = json.load(r)
    files = meta["data"]["latestVersion"]["files"]
    return [(f["dataFile"]["id"], f["dataFile"]["filename"], f["dataFile"].get("filesize")) for f in files]


def fetch(item, out: Path, retries=4):
    fid, name, size = item
    dst = out / name
    if dst.exists() and (size is None or dst.stat().st_size == size):
        return name, "cached"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{API}/access/datafile/{fid}", timeout=120) as r:
                data = r.read()
            if size is not None and len(data) != size:
                raise IOError(f"size mismatch {len(data)} != {size}")
            dst.write_bytes(data)
            return name, "ok"
        except Exception as e:  # noqa: BLE001
            err = e
    return name, f"FAILED: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/TESS")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = list_files()
    print(f"{len(items)} files listed")
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda it: fetch(it, out), items))
    failed = [r for r in res if r[1].startswith("FAILED")]
    print(f"done: {len(res) - len(failed)} ok, {len(failed)} failed")
    for f in failed[:20]:
        print(f)
    (out / "_manifest.json").write_text(json.dumps([{"id": i, "file": n, "size": s} for i, n, s in items], indent=1))


if __name__ == "__main__":
    main()
