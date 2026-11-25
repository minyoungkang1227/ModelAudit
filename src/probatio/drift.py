"""입력 분포가 바뀌었는지 본다(PSI).

PSI는 기준 표본과 새 표본의 구간별 비율 p, q로 sum((p - q) * ln(p / q))를 낸 값이다.
업계 관행은 0.10 이상 주의, 0.25 이상 실패다. 그런데 표본이 작으면 분포가 그대로여도 PSI가 0이 아니다.
분포가 같을 때 PSI / (1/n + 1/m)은 자유도 (구간 수 - 1)인 카이제곱 분포를 근사적으로 따른다.
그래서 관행 기준에 더해 '우연으로 나올 수 있는 크기인가'를 함께 본다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

WARN, FAIL = 0.10, 0.25


def _counts(ref: pd.Series, cur: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """두 표본을 같은 구간으로 나눠 센다. 숫자는 기준 표본의 분위수 구간, 그 밖은 범주 그대로."""
    ref, cur = ref.dropna(), cur.dropna()
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > bins:
        edges = np.unique(np.quantile(ref.to_numpy(float), np.linspace(0, 1, bins + 1)))
        if len(edges) >= 3:
            edges[0], edges[-1] = -np.inf, np.inf
            return (np.histogram(ref.to_numpy(float), edges)[0], np.histogram(cur.to_numpy(float), edges)[0])
    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    r = ref.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    c = cur.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    return r, c


def psi_from_counts(r: np.ndarray, c: np.ndarray, eps: float = 0.5) -> float:
    """빈 구간에서 ln(0)이 나오지 않도록 구간마다 eps를 더한 뒤 계산한다."""
    k = len(r)
    p = (r + eps) / (r.sum() + eps * k)
    q = (c + eps) / (c.sum() + eps * k)
    return float(np.sum((p - q) * np.log(p / q)))


@dataclass
class Drift:
    feature: str
    psi: float
    p_value: float          # 분포가 같을 때 이만한 PSI가 나올 확률(카이제곱 근사)
    bins: int
    n_ref: int
    n_cur: int
    p_adjusted: float = float("nan")   # 여러 변수를 한꺼번에 본 것을 감안한 값(Holm)

    def status(self, adjusted: bool = True) -> str:
        """adjusted=False면 업계 관행(PSI 크기만), True면 우연이 아닐 때만 올린다."""
        if adjusted and not (self.p_adjusted < 0.05):
            return "통과"
        return "실패" if self.psi >= FAIL else "주의" if self.psi >= WARN else "통과"


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10, name: str = "") -> Drift:
    r, c = _counts(ref, cur, bins)
    value = psi_from_counts(r, c)
    n, m, k = int(r.sum()), int(c.sum()), len(r)
    p = float(stats.chi2.sf(value / (1 / n + 1 / m), k - 1)) if k > 1 and n and m else float("nan")
    return Drift(name or str(ref.name), value, p, k, n, m)


def holm(p_values: list[float]) -> list[float]:
    """여러 검정을 한꺼번에 할 때 하나라도 우연히 걸릴 확률을 5%로 묶는 보정."""
    order = np.argsort(p_values)
    out, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, p_values[i] * (len(p_values) - rank)))
        out[i] = running
    return out


def check(ref: pd.DataFrame, cur: pd.DataFrame, columns: list[str] | None = None, bins: int = 10) -> list[Drift]:
    """변수마다 PSI를 내고, 변수 개수만큼 본 것을 감안해 보정한다. 점수 열도 같은 방식으로 넣으면 된다."""
    columns = columns or [c for c in ref.columns if c in cur.columns]
    rows = [psi(ref[c], cur[c], bins, c) for c in columns]
    for row, adj in zip(rows, holm([r.p_value for r in rows])):
        row.p_adjusted = adj
    return rows
