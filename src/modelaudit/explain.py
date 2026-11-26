"""모델이 내놓는 거절 사유가 믿을 만한지 본다. 모델 내부는 보지 않는다.

검증 대상은 '사유를 말해 주는 함수'다. 거절된 사람마다 가장 큰 사유로 변수 하나를 돌려준다고 본다.

1. 충실도: 말한 사유가 실제로 점수를 움직이는가. 변수를 하나씩 평범한 값(기준 표본의 중앙값·최빈값)으로 바꿔 넣고
   점수가 얼마나 오르는지 잰다. 말한 사유가 그 순위에서 상위 3개 안에 들면 맞는 것으로 센다.
2. 일관성: 거의 같은 사람에게 같은 사유를 말하는가. 숫자 변수를 1% 안에서 흔든 뒤 사유가 그대로인 비율.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .fairness import Model

Explainer = Callable[[pd.DataFrame], np.ndarray]   # 표를 받아 사람마다 사유(변수 이름) 하나를 돌려주는 함수
FIDELITY_WARN, FIDELITY_FAIL = 0.9, 0.5            # 이 도구가 정한 값이다. 처음엔 0.8이었으나 거짓 사유 20%를 놓쳐 0.9로 올렸다
CONSIST_WARN, CONSIST_FAIL = 0.95, 0.8


@dataclass
class Explained:
    fidelity: float       # 말한 사유가 실제 영향 상위 3개 안에 든 비율
    chance: float         # 아무 변수나 말했을 때 기대되는 비율
    consistency: float    # 입력을 1% 안에서 흔들어도 사유가 그대로인 비율
    n: int

    @property
    def fidelity_status(self) -> str:
        return "실패" if self.fidelity < FIDELITY_FAIL else "주의" if self.fidelity < FIDELITY_WARN else "통과"

    @property
    def consistency_status(self) -> str:
        return "실패" if self.consistency < CONSIST_FAIL else "주의" if self.consistency < CONSIST_WARN else "통과"

    @property
    def status(self) -> str:
        order = ["통과", "주의", "실패"]
        return max(self.fidelity_status, self.consistency_status, key=order.index)


def reference_values(ref: pd.DataFrame) -> dict:
    return {c: (ref[c].median() if pd.api.types.is_numeric_dtype(ref[c]) else ref[c].mode(dropna=True).iloc[0]) for c in ref.columns}


def check(model: Model, explainer: Explainer, X: pd.DataFrame, cutoff: float, ref: pd.DataFrame,
          n: int = 3000, top: int = 3, seed: int = 0) -> Explained:
    rng = np.random.default_rng(seed)
    base_all = model(X)
    rejected = np.flatnonzero(base_all < cutoff)
    if len(rejected) == 0:
        return Explained(float("nan"), float("nan"), float("nan"), 0)
    Xs = X.iloc[rng.choice(rejected, min(n, len(rejected)), replace=False)].reset_index(drop=True)
    base, stated = model(Xs), np.asarray(explainer(Xs), dtype=object)
    values, cols = reference_values(ref), list(X.columns)
    gain = np.column_stack([model(Xs.assign(**{c: values[c]})) - base for c in cols])   # 평범한 값으로 바꿨을 때 오르는 점수
    rank = (-gain).argsort(axis=1).argsort(axis=1)                                      # 0이 가장 큰 영향
    idx = np.array([cols.index(s) if s in cols else -1 for s in stated])
    rows = np.arange(len(Xs))
    ok = (idx >= 0) & (rank[rows, idx] < top) & (gain[rows, idx] > 0)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(Xs[c])]
    shaken = Xs.assign(**{c: Xs[c] * (1 + rng.uniform(-0.01, 0.01, len(Xs))) for c in num})
    same = np.asarray(explainer(shaken), dtype=object) == stated
    return Explained(float(ok.mean()), top / len(cols), float(same.mean()), len(Xs))
