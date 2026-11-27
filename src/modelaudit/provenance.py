"""자료의 출처 기록. 보고서가 어느 판본의 자료로 나왔는지 나중에 증명할 수 있게 한다.

자료 CSV 옆에 <이름>.meta.json을 둔다. 내려받거나 가공한 스크립트가 쓰고, 보고서가 읽어 해시를 다시 잰다.
BCBS 239 원칙 3(정확성·무결성)과 원칙 5(적시성: 기준일)를 이 도구의 입력 자료에 적용한 것이다.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta_path(csv: str | Path) -> Path:
    csv = Path(csv)
    return csv.with_name(csv.stem + ".meta.json")


def write(csv: str | Path, *, source: str, url: str, rows: int, raw_sha256: str | None = None, steps: str = "") -> Path:
    """가공이 끝난 CSV의 출처·받은 날·해시를 적는다."""
    meta = {"source": source, "url": url, "fetched_at": datetime.date.today().isoformat(), "rows": rows,
            "raw_sha256": raw_sha256, "csv_sha256": sha256(csv), "steps": steps}
    out = meta_path(csv)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def check(csv: str | Path) -> tuple[str, str]:
    """(상태, 보고서에 적을 문장). 상태는 ok / missing / changed."""
    m = meta_path(csv)
    if not m.exists():
        return "missing", f"자료 출처 기록({m.name})이 없다. 이 보고서가 어느 판본의 자료로 나왔는지 증명할 수 없다."
    meta = json.loads(m.read_text(encoding="utf-8"))
    now = sha256(csv)
    if now != meta.get("csv_sha256"):
        return "changed", (f"자료 파일이 출처 기록 이후 바뀌었다(기록 {meta.get('csv_sha256', '')[:12]}, 지금 {now[:12]}). "
                           "손으로 고쳤다면 그 이유를 남기고 기록을 다시 만든다.")
    return "ok", (f"자료: {meta['source']} ({meta['url']}), 받은 날 {meta['fetched_at']}, {meta['rows']}행, "
                  f"SHA-256 {now[:12]}. 가공: {meta.get('steps') or '적지 않음'}.")
