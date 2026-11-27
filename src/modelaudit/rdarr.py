"""저장소 자체 점검: BCBS 239가 든 '비효과적 위험 데이터 집계·보고'의 10가지 징후.

출처: FRM Part II Reading 7, "In contrast, ineffective risk data aggregation and reporting may include ..."
(BCBS 239 이행 점검 보고서의 목록을 교재가 옮긴 것).

이 점검은 판정이 아니라 **증거 찾기**다. 저장소의 코드·문서에서 각 징후를 막는 장치가 있는지 흔적을 찾고,
찾은 파일을 근거로 적는다. 흔적이 있다고 장치가 제대로 돈다는 뜻은 아니고, 흔적이 없다고 장치가 없다는
뜻도 아니다(다른 이름으로 있을 수 있다). 사람이 근거 파일을 열어 확인하는 것까지가 점검이다.

  python -m modelaudit selfcheck --repo ../collatio
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CODE = {".py", ".yml", ".yaml", ".json", ".toml", ".cfg", ".ini", ".bat", ".sh"}
TEXT = {".py", ".md", ".yml", ".yaml", ".json", ".toml", ".txt", ".cfg", ".ini", ".bat", ".sh"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "runs", "dist", "build", ".pytest_cache"}
MAX_BYTES = 400_000


@dataclass(frozen=True)
class Item:
    no: int
    symptom: str          # 교재의 징후
    guard: str            # 그 징후를 막는 장치(우리가 찾는 것)
    content: str = ""     # 파일 내용에서 찾을 정규식
    names: str = ""       # 파일 경로에서 찾을 정규식
    min_hits: int = 1
    code_only: bool = True  # 내용은 코드·설정 파일에서만 찾는다(문서에 단어만 적은 것은 세지 않는다)


ITEMS = (
    Item(1, "데이터 품질 관리 비효율", "입력 자료를 검사하는 코드",
         content=r"def (validate|check)\w*|quality|품질 점검"),
    Item(2, "품질 규칙 부실(최소 기준 없음)", "명시된 기준값·스키마",
         content=r"\bschema\b|THRESHOLD|thresholds?\b|Field\(ge=|rel_tol|min_per_cell|STALE_DAYS|\bRANGES\b|\bCHOICES\b|\bLIMITS\b"),
    Item(3, "감독(oversight) 부재", "검수·승인 기록",
         content=r"\breviewer\b|approve|검수|checked_by", names=r"CODEOWNERS$", code_only=False),
    Item(4, "에스컬레이션 체계 부재", "문제가 생기면 자동으로 멈추거나 알리는 장치",
         content=r"sys\.exit\(1\)|SystemExit\(1|return 1 if|telegram|send_alert|notify", names=r"\.github/workflows/.+\.ya?ml$"),
    Item(5, "품질 관리 약점", "자동 테스트(10개 이상)", content=r"^\s*def test_",
         names=r"(^|/)(test_\w+|_smoke_test\w*|\w+_test)\.py$", min_hits=10),
    Item(6, "문서화 안 된 수작업 과다", "수작업 수정의 사유 기록",
         content=r"changes\.yaml|[\"']changes[\"']|reason=|수정 사유|변경 사유", names=r"CHANGELOG"),
    Item(7, "핵심 리포트 간 대사 부재", "서로 다른 산출물이 맞는지 맞춰 보는 검사",
         content=r"reconcil|대사|대조|어긋나|맞지 않|digest|matches"),
    Item(8, "차이 분석(variance analysis) 부재", "기준 대비 변화·차이를 재는 코드",
         content=r"\bvariance\b|\bdrift\b|\bpsi\b|PSI|pairwise|차이 구간|전후 비교|backtest|예측 vs 실제|track_record"),
    Item(9, "하위 데이터 적시 수집 불가", "자료의 기준일·신선도 확인",
         content=r"\bas_?of\b|freshness|\bstale\b|STALE|기준일|fetched_at|due_status", code_only=False),
    Item(10, "기준 데이터 표준화 부재", "데이터 사전·스키마 정의",
         names=r"(data_dictionary|schema|SOURCES|sources)\.(md|py)$|dictionary"),
)


@dataclass
class Result:
    item: Item
    hits: int = 0
    files: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.hits >= self.item.min_hits


SELF = Path(__file__).resolve()


def _files(repo: Path):
    for p in sorted(repo.rglob("*")):
        if p.resolve() == SELF:      # 이 점검 코드의 정규식 문자열이 스스로를 흔적으로 잡지 않게 한다
            continue
        rel = p.relative_to(repo)
        if p.is_file() and not (set(rel.parts[:-1]) & SKIP_DIRS) and p.suffix.lower() in TEXT | {""}:
            yield p, rel.as_posix()


def check(repo: str | Path) -> list[Result]:
    repo = Path(repo)
    if not repo.is_dir():
        raise ValueError(f"폴더가 아니다: {repo}")
    results = [Result(it) for it in ITEMS]
    pats = [(re.compile(it.content, re.M) if it.content else None, re.compile(it.names) if it.names else None) for it in ITEMS]
    for path, rel in _files(repo):
        text = None
        for r, (cpat, npat) in zip(results, pats):
            n = 0
            if npat and npat.search(rel):
                n += 1
            if cpat and path.suffix and (path.suffix.lower() in CODE or not r.item.code_only):
                if text is None:
                    try:
                        text = path.read_text(encoding="utf-8")[:MAX_BYTES] if path.stat().st_size <= MAX_BYTES * 4 else ""
                    except (UnicodeDecodeError, OSError):
                        text = ""
                n += len(cpat.findall(text))
            if n:
                r.hits += n
                r.files.append(rel)
    return results


def render(repo: str | Path, results: list[Result]) -> str:
    name = Path(repo).resolve().name
    n = sum(r.found for r in results)
    lines = [f"# 데이터 집계·보고 자체 점검: {name}", "",
             "기준: BCBS 239가 든 비효과적 집계·보고의 10가지 징후(FRM Part II Reading 7). "
             "각 징후를 막는 장치의 **흔적**을 코드와 문서에서 찾았다. 판정이 아니다. 근거 파일을 열어 사람이 확인한다.", "",
             f"흔적 있음 {n}/10", "",
             "| # | 징후 | 막는 장치 | 흔적 | 근거 파일(앞 3개) |", "|---|---|---|---|---|"]
    for r in results:
        mark = "있음" if r.found else ("부족" if r.hits else "없음")
        if r.item.min_hits > 1:
            mark += f" ({r.hits}개)"
        files = ", ".join(f"`{f}`" for f in r.files[:3]) or "-"
        lines.append(f"| {r.item.no} | {r.item.symptom} | {r.item.guard} | {mark} | {files} |")
    missing = [r.item for r in results if not r.found]
    if missing:
        lines += ["", "## 흔적이 없는 것", ""] + [f"- {it.no}. {it.symptom}: {it.guard}를 찾지 못했다." for it in missing]
    lines += ["", "## 이 점검이 못 하는 것", "",
              "- 장치가 실제로 도는지(테스트가 통과하는지, CI가 켜져 있는지)는 보지 않는다.",
              "- 이름이 다른 장치는 놓친다. 흔적 없음이 곧 장치 없음은 아니다.",
              "- 문서에 '대조'라고 적기만 해도 흔적으로 잡힌다. 근거 파일을 열어 본다."]
    return "\n".join(lines) + "\n"
