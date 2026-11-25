"""문제를 심고 점검이 잡는지 센다. 심지 않은 경우(대조)에서 잘못 잡는 비율도 함께 센다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import drift, fairness, samples


@dataclass
class Setup:
    """기본값은 German Credit(UCI Statlog) 열 이름이다."""
    label: str = "credit_risk"          # 1 = 우량
    group_column: str = "age"
    group_edge: float = 25              # 이 값 이하를 한 집단으로 본다
    flip_values: tuple = (22, 40)       # 바꿔 넣기에 쓰는 두 값
    exclude: tuple = ("age", "personal_status_sex", "foreign_worker")   # 표본 모델에 넣지 않는 보호 속성
    numeric: str = "amount"
    categorical: str = "housing"
    category: str = "rent"
    approve_share: float = 0.7
    ref_share: float = 0.6

    def member(self, X: pd.DataFrame):
        return X[self.group_column] <= self.group_edge

    def group(self, X: pd.DataFrame):
        return np.where(self.member(X), f"{self.group_edge:g} 이하", f"{self.group_edge:g} 초과")


def _split(df: pd.DataFrame, share: float, rng):
    idx = rng.permutation(len(df))
    k = int(share * len(df))
    return df.iloc[idx[:k]].reset_index(drop=True), df.iloc[idx[k:]].reset_index(drop=True)


def drift_trial(df: pd.DataFrame, s: Setup, seeds: int = 30,
                factors=(1.1, 1.3, 1.6), weights=(1.5, 2.5, 4.0)) -> pd.DataFrame:
    """대조(아무것도 안 바꿈)와 심은 경우를 관행 기준과 보정 기준으로 각각 판정한다."""
    X = df.drop(columns=[s.label])
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ref, cur = _split(X, s.ref_share, rng)
        res = drift.check(ref, cur)
        for rule, adj in (("관행", False), ("보정", True)):
            rows.append(dict(kind="대조", level="없음", rule=rule, seed=seed,
                             flagged=any(r.status(adj) != "통과" for r in res)))
        cases = [("금액 배율", f, s.numeric, samples.scaled(cur, s.numeric, f)) for f in factors]
        cases += [("구성 변화", w, s.categorical, samples.remixed(cur, s.categorical, s.category, w, rng)) for w in weights]
        for kind, level, col, changed in cases:
            hit = next(r for r in drift.check(ref, changed) if r.feature == col)
            for rule, adj in (("관행", False), ("보정", True)):
                rows.append(dict(kind=kind, level=level, rule=rule, seed=seed, flagged=hit.status(adj) != "통과"))
    return pd.DataFrame(rows)


def fairness_trial(df: pd.DataFrame, s: Setup, seeds: int = 30,
                   deltas=(0.05, 0.10, 0.20), shares=(0.2, 0.4, 0.6)) -> pd.DataFrame:
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        train, test = _split(df, s.ref_share, rng)
        y_train, y_test = train[s.label].to_numpy(), test[s.label].to_numpy()
        Xtr, Xte = train.drop(columns=[s.label]), test.drop(columns=[s.label])
        base = samples.train(Xtr, y_train, s.exclude)
        cases = [("대조", "없음", base, base)]
        cases += [("직접 차별", d, samples.with_penalty(base, s.member, d), base) for d in deltas]
        for share in shares:
            m = samples.train(Xtr, samples.biased_labels(y_train, s.member(Xtr), share, rng), s.exclude)
            cases.append(("간접 차별", share, m, m))
        for kind, level, model, cutoff_model in cases:
            cut = samples.cutoff_for(cutoff_model(Xtr), s.approve_share)   # 기준 점수는 문제를 심기 전 점수로 정한다
            approved = model(Xte) >= cut
            checks = {
                "승인율 비": fairness.disparity(approved, s.group(Xte)).status,
                "기회 격차": fairness.opportunity(approved, y_test, s.group(Xte)).status,
                "바꿔 넣기": fairness.flip(model, Xte, s.group_column, *s.flip_values, cut).status,
            }
            for check, status in checks.items():
                rows.append(dict(kind=kind, level=level, check=check, seed=seed, flagged=status != "통과", failed=status == "실패"))
    return pd.DataFrame(rows)


def summarize(trials: pd.DataFrame, by: str) -> pd.DataFrame:
    """심은 종류·세기별로 잡은 비율."""
    out = trials.groupby(["kind", "level", by], sort=False)["flagged"].agg(["sum", "count"]).reset_index()
    out["rate"] = out["sum"] / out["count"]
    return out
