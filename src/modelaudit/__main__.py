"""python -m modelaudit report --data data/german.csv   : 표본 모델 하나를 점검해 보고서를 낸다
python -m modelaudit planted --data data/german.csv  : 문제를 심고 몇 개를 잡는지 센다"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import drift, fairness, planted, report, samples


def cmd_report(df, s, out):
    rng = np.random.default_rng(0)
    ref, cur = planted._split(df, s.ref_share, rng)
    Xr, Xc = ref.drop(columns=[s.label]), cur.drop(columns=[s.label])
    model = samples.train(Xr, ref[s.label], s.exclude)
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
    text = report.build("표본 모델(로지스틱 회귀)", drifts, fair, notes)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


def cmd_planted(df, s, out, seeds):
    d = planted.drift_trial(df, s, seeds)
    f = planted.fairness_trial(df, s, seeds)
    d.to_csv(out / "planted_drift.csv", index=False)
    f.to_csv(out / "planted_fairness.csv", index=False)
    text = (f"# 심은 문제 점검 ({seeds}회 반복, 칸은 잡은 횟수/시도)\n\n## 드리프트\n\n"
            + report.trial_table(planted.summarize(d, "rule"), "rule")
            + "\n## 공정성\n\n" + report.trial_table(planted.summarize(f, "check"), "check"))
    (out / "planted.md").write_text(text, encoding="utf-8")
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="modelaudit")
    ap.add_argument("command", choices=["report", "planted"])
    ap.add_argument("--data", required=True, help="정답 열이 있는 CSV")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--preset", choices=sorted(planted.PRESETS), default="german", help="자료와 집단 설정")
    a = ap.parse_args(argv)
    df, s = pd.read_csv(a.data), planted.PRESETS[a.preset]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cmd_report(df, s, out) if a.command == "report" else cmd_planted(df, s, out, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
