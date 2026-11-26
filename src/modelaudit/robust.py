"""입력을 아주 조금 바꿨을 때 승인·거절이 뒤집히는지 본다.

숫자 변수를 하나씩 1% 키우고 1% 줄여 모델에 다시 넣는다. 금액이 1% 다르다고 결과가 바뀌는 사람이 많으면
모델이 기준선 근처에서 불안정한 것이다. 기준선 바로 옆에 있는 사람은 어느 모델에서나 뒤집히므로 0이 되지는 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fairness import Model

WARN, FAIL = 0.02, 0.05   # 이 도구가 정한 값이다. 근거 규정은 없다


@dataclass
class Sensitivity:
    column: str
    flipped: float      # 1% 키우거나 줄였을 때 승인·거절이 바뀐 사람의 비율
    mean_shift: float   # 점수가 움직인 크기의 평균

    @property
    def status(self) -> str:
        return "실패" if self.flipped >= FAIL else "주의" if self.flipped >= WARN else "통과"


def check(model: Model, X: pd.DataFrame, cutoff: float, eps: float = 0.01, columns: list[str] | None = None) -> list[Sensitivity]:
    columns = columns or [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 10]
    base = model(X)
    decided = base >= cutoff
    out = []
    for c in columns:
        up, down = model(X.assign(**{c: X[c] * (1 + eps)})), model(X.assign(**{c: X[c] * (1 - eps)}))
        flipped = ((up >= cutoff) != decided) | ((down >= cutoff) != decided)
        out.append(Sensitivity(c, float(flipped.mean()), float(np.mean((np.abs(up - base) + np.abs(down - base)) / 2))))
    return out
