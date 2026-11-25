"""점검 결과를 검증 보고서(마크다운)로 묶는다."""
from __future__ import annotations

import pandas as pd

from . import drift, fairness

ORDER = {"통과": 0, "관찰": 1, "주의": 2, "실패": 3}

# 점검 항목이 닿는 규제 항목. 출처는 docs/mapping.md에 적는다
MAPPING = {
    "드리프트": "금융위 가이드라인 ④ 신뢰성(모델 성능·데이터 품질의 정기 점검) / 금감원 AI RMF 위험통제(운영 단계 모니터링)",
    "공정성": "금융위 가이드라인 ④ 신뢰성(편향성 점검) / 금감원 AI RMF 위험평가",
}


def overall(statuses: list[str]) -> str:
    return max(statuses, key=ORDER.get) if statuses else "검사 안 함"


def build(title: str, drifts: list[drift.Drift], fair: list, notes: list[str] | None = None) -> str:
    d_status = overall([d.status() for d in drifts])
    f_status = overall([f.status for f in fair])
    lines = [f"# 검증 보고서: {title}", "", "## 요약", "", "| 영역 | 판정 | 닿는 규제 항목 |", "|---|---|---|",
             f"| 드리프트 | {d_status} | {MAPPING['드리프트']} |", f"| 공정성 | {f_status} | {MAPPING['공정성']} |", ""]
    lines += ["## 드리프트", "", "PSI 0.10 이상 주의, 0.25 이상 실패. 우연으로 나올 수 있는 크기면 올리지 않는다(보정 p < 0.05). 우연은 아니지만 0.10 미만이면 관찰.", "",
              "| 변수 | PSI | 보정 p | 관행 판정 | 판정 |", "|---|---|---|---|---|"]
    for d in sorted(drifts, key=lambda r: -r.psi):
        lines.append(f"| {d.feature} | {d.psi:.3f} | {d.p_adjusted:.3f} | {d.status(False)} | {d.status()} |")
    lines += ["", "## 공정성", "", "| 점검 | 값 | 판정 |", "|---|---|---|"]
    for f in fair:
        if isinstance(f, fairness.Versus):
            lines.append(f"| 기준 모델 비교 | 집단 안 {f.in_group:+.1%}, 집단 밖 {f.out_group:+.1%}, 차이 {f.extra:+.1%}, p {f.p_value:.3f} (n={f.n_in}) | {f.status} |")
        elif isinstance(f, fairness.Disparity):
            rates = ", ".join(f"{g} {r:.1%} (n={f.counts[g]})" for g, r in f.rates.items())
            lines.append(f"| {f.name} | 비 {f.ratio:.2f}, p {f.p_value:.3f}. {rates} | {f.status} |")
        else:
            lines.append(f"| 바꿔 넣기({f.column}) | 결과가 바뀐 비율 {f.changed:.1%}, 점수 변화 평균 {f.mean_shift:.4f} | {f.status} |")
    if notes:
        lines += ["", "## 읽을 때 주의", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def trial_table(summary: pd.DataFrame, by: str) -> str:
    wide = summary.assign(cell=lambda t: t["sum"].astype(str) + "/" + t["count"].astype(str)) \
                  .pivot_table(index=["kind", "level"], columns=by, values="cell", aggfunc="first", sort=False)
    head = "| 심은 것 | 세기 | " + " | ".join(wide.columns) + " |"
    rows = [f"| {k} | {l} | " + " | ".join(wide.loc[(k, l)]) + " |" for k, l in wide.index]
    return "\n".join([head, "|---" * (len(wide.columns) + 2) + "|"] + rows) + "\n"
