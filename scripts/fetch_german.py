"""UCI에서 German Credit(Statlog)을 받아 data/german.csv로 저장한다. 표준 라이브러리만 쓴다.

    python scripts/fetch_german.py                      # UCI에서 받는다
    python scripts/fetch_german.py 받은폴더/german.data   # 이미 받은 파일을 바꾼다
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hashlib  # noqa: E402

from modelaudit import provenance  # noqa: E402
from modelaudit.data import parse_german  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_bytes()
        text = raw.decode("ascii")
    else:
        raw = urllib.request.urlopen(URL, timeout=60).read()
        raw = zipfile.ZipFile(io.BytesIO(raw)).read("german.data")
        text = raw.decode("ascii")
    df = parse_german(text)
    out = Path(__file__).resolve().parents[1] / "data" / "german.csv"
    df.to_csv(out, index=False)
    provenance.write(out, source="UCI Statlog German Credit", url=URL, rows=len(df),
                     raw_sha256=hashlib.sha256(raw).hexdigest(),
                     steps="정답 1/2 → 1 우량/0 부실, 주거 형태 부호를 말로 풀기")
    print(f"{len(df)}행, 우량 {int(df.credit_risk.sum())}건 -> {out}")
