"""UCI에서 German Credit(Statlog)을 받아 data/german.csv로 저장한다. 표준 라이브러리만 쓴다.

    python scripts/fetch_german.py
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from probatio.data import parse_german  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"

if __name__ == "__main__":
    raw = urllib.request.urlopen(URL, timeout=60).read()
    text = zipfile.ZipFile(io.BytesIO(raw)).read("german.data").decode("ascii")
    df = parse_german(text)
    out = Path(__file__).resolve().parents[1] / "data" / "german.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)}행, 우량 {int(df.credit_risk.sum())}건 -> {out}")
