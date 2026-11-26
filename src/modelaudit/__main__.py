"""python -m modelaudit report --data data/german.csv   : 표본 모델 하나를 점검해 보고서를 낸다
python -m modelaudit planted --data data/german.csv  : 문제를 심고 몇 개를 잡는지 센다"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import json

from . import drift, explain, fairness, llm, perf, planted, report, robust, samples


def cmd_report(df, s, out, risk_score=None):
    rng = np.random.default_rng(0)
    ref, cur = planted._split(df, s.ref_share, rng)
    Xr, Xc = ref.drop(columns=[s.label]), cur.drop(columns=[s.label])
    model, explainer = samples.train_explained(Xr, ref[s.label], s.exclude)
    cut = samples.cutoff_for(model(Xr), s.approve_share)
    scores = pd.DataFrame({"(모델 점수)": model(Xr)}), pd.DataFrame({"(모델 점수)": model(Xc)})
    drifts = drift.check(pd.concat([Xr, scores[0]], axis=1), pd.concat([Xc, scores[1]], axis=1))
    approved = model(Xc) >= cut
    fair = [fairness.disparity(approved, s.group(Xc)), fairness.opportunity(approved, cur[s.label], s.group(Xc)),
            fairness.flip(model, Xc, s.group_column, *s.flip_values, cut)]
    bench = samples.train_benchmark(Xr, ref[s.label], s.exclude)
    fair.append(fairness.versus(approved, bench(Xc) >= samples.cutoff_for(bench(Xr), s.approve_share), s.member(Xc)))
    notes = ["기준 표본과 새 표본은 같은 자료를 무작위로 나눈 것이다. 실제 시점 차이가 아니다.",
             "승인율 차이는 집단의 실제 부실률 차이에서도 생긴다. 기회 격차와 함께 읽는다.",
             f"표본 모델은 {', '.join(s.exclude)}를 입력으로 받지 않는다."]
    notes.append("판정 기준 가운데 PSI 0.10·0.25와 승인율 비 0.8은 관행을 빌린 것이고, AUC 하락 0.03·0.05와 기준 모델 차이 2%p·5%p는 이 도구가 정한 값이다.")
    notes.append("기준 표본의 AUC는 모델이 배우지 않은 조각에서 잰 값이다(5조각 교차). 배운 자료에서 그대로 재면 부풀려져서 새 표본과 견줄 수 없다.")
    p = perf.check(ref[s.label], samples.out_of_fold(Xr, ref[s.label], s.exclude), cur[s.label], model(Xc))
    text = report.build("표본 모델(로지스틱 회귀)", drifts, fair, notes, perf=p, robustness=robust.check(model, Xc, cut),
                        explained=explain.check(model, explainer, Xc, cut, Xr), use=s.use, high_impact=s.high_impact, risk_score=risk_score)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


def cmd_planted(df, s, out, seeds):
    d = planted.drift_trial(df, s, seeds)
    f = planted.fairness_trial(df, s, seeds)
    d.to_csv(out / "planted_drift.csv", index=False)
    f.to_csv(out / "planted_fairness.csv", index=False)
    r = planted.robustness_trial(df, s, seeds)
    r.to_csv(out / "planted_robustness.csv", index=False)
    e = planted.explain_trial(df, s, seeds)
    e.to_csv(out / "planted_explain.csv", index=False)
    text = (f"# 심은 문제 점검 ({seeds}회 반복, 칸은 잡은 횟수/시도)\n\n## 드리프트\n\n"
            + report.trial_table(planted.summarize(d, "rule"), "rule")
            + "\n## 공정성\n\n" + report.trial_table(planted.summarize(f, "check"), "check")
            + "\n## 강건성\n\n" + report.trial_table(planted.summarize(r, "check"), "check")
            + "\n## 설명가능성\n\n" + report.trial_table(planted.summarize(e, "check"), "check"))
    (out / "planted.md").write_text(text, encoding="utf-8")
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="modelaudit")
    ap.add_argument("command", choices=["report", "planted", "llm"])
    ap.add_argument("--data", help="정답 열이 있는 CSV (report, planted)")
    ap.add_argument("--collatio", help="Collatio pipeline이 낸 결과 JSON (llm)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--risk-score", type=float, help="회사가 AI RMF 16개 항목으로 낸 위험점수(0~100). 이 도구는 이 점수를 내지 않는다")
    ap.add_argument("--preset", choices=sorted(planted.PRESETS), default="german", help="자료와 집단 설정")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.command == "llm":
        if not a.collatio:
            ap.error("llm에는 --collatio가 필요하다")
        raw = Path(a.collatio).read_text(encoding="utf-8")
        checks, mock = llm.from_collatio(json.loads(raw[raw.index("{"):]))
        text = report.build_llm(checks, mock, Path(a.collatio).name)
        (out / "llm_report.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
    if not a.data:
        ap.error("--data가 필요하다")
    df, s = pd.read_csv(a.data), planted.PRESETS[a.preset]
    cmd_report(df, s, out, a.risk_score) if a.command == "report" else cmd_planted(df, s, out, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
