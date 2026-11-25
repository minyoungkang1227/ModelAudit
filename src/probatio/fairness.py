"""집단에 따라 결과가 갈리는지 본다. 세 가지를 따로 잰다.

1. 승인율 비: 집단별 승인율의 최솟값 / 최댓값. 0.8 미만이면 문제로 보는 관행(미국 고용 분야의 5분의 4 규칙)을 빌렸다.
2. 기회 격차: 실제로 우량한 사람만 놓고 본 승인율 차이. 집단의 실제 부실률이 달라서 생기는 차이를 걷어낸다.
3. 바꿔 넣기: 다른 값은 그대로 두고 집단 값만 바꿔 모델에 다시 넣는다. 결과가 달라지면 모델이 그 값을 직접 쓴 것이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

Model = Callable[[pd.DataFrame], np.ndarray]   # 표를 받아 '우량할 확률'을 돌려주는 함수. 내부는 보지 않는다


@dataclass
class Disparity:
    name: str
    rates: dict[str, float]
    counts: dict[str, int]
    ratio: float
    gap: float
    p_value: float

    @property
    def status(self) -> str:
        significant = self.p_value < 0.05
        if self.ratio < 0.8 and significant:
            return "실패"
        if self.ratio < 0.8 or (significant and self.ratio < 0.9):
            return "주의"   # 0.8 미만인데 유의하지 않으면 표본이 작아 판단을 못 한 것이다
        return "통과"


def disparity(approved, group, name: str = "승인율 비") -> Disparity:
    approved, group = np.asarray(approved, bool), pd.Series(np.asarray(group)).astype(str)
    table = pd.crosstab(group, approved).reindex(columns=[False, True], fill_value=0)
    rates = (table[True] / table.sum(axis=1)).to_dict()
    counts = table.sum(axis=1).astype(int).to_dict()
    usable = table.loc[:, table.sum(axis=0) > 0]
    p = float(stats.chi2_contingency(usable)[1]) if usable.shape[1] == 2 and len(usable) > 1 else 1.0
    hi, lo = max(rates.values()), min(rates.values())
    return Disparity(name, rates, counts, lo / hi if hi > 0 else 1.0, hi - lo, p)


def opportunity(approved, y, group) -> Disparity:
    """실제로 우량했던 사람(y == 1)만 놓고 본 승인율 차이."""
    keep = np.asarray(y) == 1
    return disparity(np.asarray(approved)[keep], np.asarray(group)[keep], "기회 격차")


@dataclass
class Flip:
    column: str
    changed: float      # 승인/거절이 뒤바뀐 사람의 비율
    mean_shift: float   # 점수가 움직인 크기의 평균

    @property
    def status(self) -> str:
        return "실패" if self.changed >= 0.01 else "주의" if self.mean_shift > 1e-9 else "통과"


def flip(model: Model, X: pd.DataFrame, column: str, a, b, cutoff: float) -> Flip:
    """column 값만 a와 b로 바꿔 넣어 점수를 두 번 받는다. 모델에 그 열이 안 들어가면 차이는 0이다."""
    sa, sb = model(X.assign(**{column: a})), model(X.assign(**{column: b}))
    return Flip(column, float(np.mean((sa >= cutoff) != (sb >= cutoff))), float(np.mean(np.abs(sa - sb))))
