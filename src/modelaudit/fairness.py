"""집단에 따라 결과가 갈리는지 본다. 세 가지를 따로 잰다.

1. 승인율 비: 집단별 승인율의 최솟값 / 최댓값. 0.8 미만이면 문제로 보는 관행(미국 고용 분야의 5분의 4 규칙)을 빌렸다.
2. 기회 격차: 실제로 우량한 사람만 놓고 본 승인율 차이. 집단의 실제 부실률이 달라서 생기는 차이를 걷어낸다.
3. 바꿔 넣기: 다른 값은 그대로 두고 집단 값만 바꿔 모델에 다시 넣는다. 결과가 달라지면 모델이 그 값을 직접 쓴 것이다.
4. 기준 모델 비교: 검증자가 같은 자료로 따로 세운 모델과 나란히 놓는다. 1, 2는 자료에 원래 있던 차이와
   모델이 더한 차이를 나누지 못한다. 기준 모델도 같은 자료에서 배웠으므로 둘의 차이는 검증 대상 모델이 더한 몫이다.
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


@dataclass
class Versus:
    """검증 대상 모델과 기준 모델을 같은 사람들에게 적용해 비교한 결과."""
    in_group: float     # 집단 안에서 (대상 승인율 - 기준 승인율)
    out_group: float    # 집단 밖에서 같은 값
    p_value: float
    n_in: int

    @property
    def extra(self) -> float:
        """음수면 대상 모델이 기준 모델보다 이 집단을 그만큼 덜 승인한다(집단 밖의 변화를 뺀 값)."""
        return self.in_group - self.out_group

    @property
    def status(self) -> str:
        # 0.05, 0.02는 이 도구가 정한 값이다. 근거 규정은 없다
        if not (self.p_value < 0.05):
            return "통과"
        return "실패" if abs(self.extra) >= 0.05 else "주의" if abs(self.extra) >= 0.02 else "통과"


def versus(approved, benchmark_approved, member) -> Versus:
    """사람마다 (대상 결정 - 기준 결정)을 내고, 그 평균이 집단 안팎에서 다른지 본다."""
    d = np.asarray(approved, float) - np.asarray(benchmark_approved, float)
    m = np.asarray(member, bool)
    a, b = d[m], d[~m]
    if len(a) < 2 or len(b) < 2:
        return Versus(float("nan"), float("nan"), float("nan"), int(m.sum()))
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    p = float(2 * stats.norm.sf(abs(a.mean() - b.mean()) / se)) if se > 0 else 1.0
    return Versus(float(a.mean()), float(b.mean()), p, int(m.sum()))
