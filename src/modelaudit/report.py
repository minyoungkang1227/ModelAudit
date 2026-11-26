"""점검 결과를 검증 보고서(마크다운)로 묶는다.

틀은 금융위 「금융분야 인공지능 가이드라인」(2026-06-22 시행)의 신뢰성 원칙과 금감원 AI RMF의 위험등급이다.
어디까지가 공개 자료에서 확인한 것이고 어디부터가 이 도구의 해석인지는 docs/mapping.md에 적는다.
"""
from __future__ import annotations

import pandas as pd

from . import drift, fairness

ORDER = {"통과": 0, "관찰": 1, "주의": 2, "실패": 3}
NOT_DONE = "검사 안 함"

# 금감원 AI RMF의 위험등급 구간(정량 점수 100점 기준)과 등급별 통제. 출처는 docs/mapping.md
GRADES = [(75, "초고위험", "AI 의사결정기구에서 출시 여부 재검토"),
          (50, "고위험", "최고의사결정기구 사전 승인과 사후 검증, 제3자 평가·검증, 운영 모니터링 강화"),
          (25, "중위험", "등급에 맞춘 통제(공개 자료에서 세부 내용을 확인하지 못함)"),
          (0, "저위험", "승인 절차 간소화 등 완화된 관리")]
HIGH_CONTROL = GRADES[1][2]


def overall(statuses: list[str]) -> str:
    done = [s for s in statuses if s in ORDER]
    return max(done, key=ORDER.get) if done else NOT_DONE


def grade(score: float | None, high_impact: bool) -> tuple[str, str, str]:
    """(등급, 근거, 요구되는 통제). 대출 심사 같은 용도는 점수와 관계없이 고위험이다. 점수는 회사가 RMF 16개 항목으로 낸 값을 받는다."""
    if score is not None:
        name, control = next((n, c) for floor, n, c in GRADES if score >= floor)
        if not high_impact or score >= 50:
            return name, f"위험점수 {score:g}점", control
    if high_impact:
        return "고위험", "대출 심사처럼 개인의 권리·의무에 중대한 영향을 미치는 용도(위험점수와 무관하게 고위험)", HIGH_CONTROL
    return "산정 안 함", "위험점수가 주어지지 않았다. 이 도구는 16개 항목 점수를 내지 않는다", "-"


def build(title: str, drifts: list[drift.Drift], fair: list, notes: list[str] | None = None, *,
          perf=None, robustness: list | None = None, explained=None, use: str = "", high_impact: bool = False, risk_score: float | None = None) -> str:
    areas = {
        "모델 성능 관리": overall(([perf.status] if perf is not None else []) + [r.status for r in robustness or []]),
        "데이터 품질": overall([d.status() for d in drifts]),
        "공정성·편향성": overall([f.status for f in fair]),
        "설명가능성": explained.status if explained is not None and explained.n else NOT_DONE,
    }
    g, why, control = grade(risk_score, high_impact)
    verdict = overall(list(areas.values()))
    missing = [a for a, s in areas.items() if s == NOT_DONE]
    lines = [f"# 검증 보고서: {title}", "", "## 1. 대상과 위험등급", "",
             "| 항목 | 내용 |", "|---|---|", f"| 용도 | {use or '적지 않음'} |",
             f"| 위험등급 (금감원 AI RMF) | {g} |", f"| 근거 | {why} |", f"| 이 등급에 요구되는 통제 | {control} |", "",
             "이 보고서는 위 통제 가운데 독립 검증에 쓸 근거를 낸다. 승인과 책임은 사람이 진다.", "",
             "## 2. 검증 의견", "", f"종합 판정: **{verdict}**" + (f" (검사하지 않은 영역: {', '.join(missing)})" if missing else ""), "",
             "신뢰성 원칙(금융위 가이드라인 7대 원칙 중 ④)의 네 영역별 판정이다.", "",
             "| 영역 | 판정 | 근거가 된 점검 |", "|---|---|---|",
             f"| 모델 성능 관리 | {areas['모델 성능 관리']} | 새 표본에서의 구별력(AUC) 유지, 입력 1% 변화에 대한 안정성 |",
             f"| 데이터 품질 | {areas['데이터 품질']} | 입력·점수 분포 변화(PSI) |",
             f"| 공정성·편향성 | {areas['공정성·편향성']} | 승인율 비, 기회 격차, 바꿔 넣기, 기준 모델 비교 |",
             f"| 설명가능성 | {areas['설명가능성']} | 거절 사유의 충실도와 일관성 |", "",
             "이 도구가 보지 않는 것: 합법성, 신의성실, 보안성 원칙과 거버넌스. 위험점수의 나머지 부분은 회사가 따로 평가해야 한다.", ""]
    if perf is not None:
        lines += ["## 3. 모델 성능", "", "| 기준 표본 AUC | 새 표본 AUC | 하락 | 하락의 95% 구간 | 판정 |", "|---|---|---|---|---|",
                  f"| {perf.auc_ref:.3f} | {perf.auc_cur:.3f} | {perf.drop:+.3f} | {perf.low:+.3f} ~ {perf.high:+.3f} | {perf.status} |", ""]
    if robustness:
        lines += ["### 강건성", "", "숫자 변수를 하나씩 1% 키우고 줄였을 때 승인·거절이 바뀐 사람의 비율이다. 2% 이상 주의, 5% 이상 실패. 아래는 큰 순서로 5개다.", "",
                  "| 변수 | 뒤집힌 비율 | 점수 변화 평균 | 판정 |", "|---|---|---|---|"]
        for r in sorted(robustness, key=lambda r: -r.flipped)[:5]:
            lines.append(f"| {r.column} | {r.flipped:.2%} | {r.mean_shift:.4f} | {r.status} |")
        lines.append("")
    flagged = [d for d in drifts if d.status() != "통과"]
    lines += ["## 4. 데이터 품질 (드리프트)", "",
              "PSI 0.10 이상 주의, 0.25 이상 실패. 우연으로 나올 수 있는 크기면 올리지 않는다(보정 p < 0.05). 우연은 아니지만 0.10 미만이면 관찰.", "",
              f"변수 {len(drifts)}개 중 {len(flagged)}개에서 변화가 잡혔다. 아래는 PSI가 큰 순서로 10개다.", "",
              "| 변수 | PSI | 보정 p | 관행 판정 | 판정 |", "|---|---|---|---|---|"]
    for d in sorted(drifts, key=lambda r: -r.psi)[:10]:
        lines.append(f"| {d.feature} | {d.psi:.3f} | {d.p_adjusted:.3f} | {d.status(False)} | {d.status()} |")
    lines += ["", "## 5. 공정성", "", "| 점검 | 값 | 판정 |", "|---|---|---|"]
    for f in fair:
        if isinstance(f, fairness.Versus):
            lines.append(f"| 기준 모델 비교 | 집단 안 {f.in_group:+.1%}, 집단 밖 {f.out_group:+.1%}, 차이 {f.extra:+.1%}, p {f.p_value:.3f} (n={f.n_in}) | {f.status} |")
        elif isinstance(f, fairness.Disparity):
            rates = ", ".join(f"{g_} {r:.1%} (n={f.counts[g_]})" for g_, r in f.rates.items())
            lines.append(f"| {f.name} | 비 {f.ratio:.2f}, p {f.p_value:.3f}. {rates} | {f.status} |")
        else:
            lines.append(f"| 바꿔 넣기({f.column}) | 결과가 바뀐 비율 {f.changed:.1%}, 점수 변화 평균 {f.mean_shift:.4f} | {f.status} |")
    if explained is not None and explained.n:
        lines += ["", "## 6. 설명가능성", "", f"거절된 사람 {explained.n}명의 거절 사유를 점검했다.", "",
                  "| 점검 | 값 | 판정 |", "|---|---|---|",
                  f"| 충실도 | 말한 사유가 실제 영향 상위 3개 안에 든 비율 {explained.fidelity:.1%} (아무 변수나 말하면 {explained.chance:.1%}) | {explained.fidelity_status} |",
                  f"| 일관성 | 입력을 1% 안에서 흔들어도 사유가 그대로인 비율 {explained.consistency:.1%} | {explained.consistency_status} |"]
    if notes:
        lines += ["", "## 7. 읽을 때 주의", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def trial_table(summary: pd.DataFrame, by: str) -> str:
    wide = summary.assign(cell=lambda t: t["sum"].astype(str) + "/" + t["count"].astype(str)) \
                  .pivot_table(index=["kind", "level"], columns=by, values="cell", aggfunc="first", sort=False)
    head = "| 심은 것 | 세기 | " + " | ".join(wide.columns) + " |"
    rows = [f"| {k} | {l} | " + " | ".join(wide.loc[(k, l)]) + " |" for k, l in wide.index]
    return "\n".join([head, "|---" * (len(wide.columns) + 2) + "|"] + rows) + "\n"


def build_llm(checks: list, mock: bool, source: str, use: str = "공시를 읽고 답하는 LLM (업무 보조)") -> str:
    """Collatio 결과를 같은 틀로 적는다. 판정은 원칙·영역별로 가장 나쁜 것을 올린다."""
    areas: dict[str, list[str]] = {}
    for c in checks:
        areas.setdefault(c.area, []).append(c.status)
    verdict = overall([c.status for c in checks])
    missing = [c.name for c in checks if c.status == NOT_DONE]
    lines = ["# 검증 보고서: LLM (Collatio 결과)", ""]
    if mock:
        lines += ["**모의 실행 결과다. 아래 숫자는 실제 모델과 무관하다.**", ""]
    lines += ["## 1. 대상", "", "| 항목 | 내용 |", "|---|---|", f"| 용도 | {use} |", f"| 근거 파일 | {source} |",
              "| 위험등급 | 산정 안 함. 용도와 회사의 위험점수에 따라 정해진다 |", "",
              "## 2. 검증 의견", "", f"종합 판정: **{verdict}**" + (f" (검사하지 않은 것: {', '.join(missing)})" if missing else ""), "",
              "| 원칙·영역 | 판정 |", "|---|---|"]
    lines += [f"| {a} | {overall(s)} |" for a, s in areas.items()]
    lines += ["", "## 3. 점검별 결과", "", "| 원칙·영역 | 점검 | 값 | 판정 |", "|---|---|---|---|"]
    lines += [f"| {c.area} | {c.name} | {c.value} | {c.status} |" for c in checks]
    lines += ["", "판정 기준 숫자는 이 도구가 정한 값이다. 점검 자체와 그 한계는 Collatio 저장소의 문서를 따른다."]
    return "\n".join(lines) + "\n"
