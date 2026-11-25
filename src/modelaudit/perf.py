"""모델이 우량과 부실을 가르는 힘(AUC)이 새 표본에서도 유지되는지 본다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

DROP_WARN, DROP_FAIL = 0.03, 0.05   # 이 도구가 정한 값이다. 근거 규정은 없다


@dataclass
class Performance:
    auc_ref: float
    auc_cur: float
    low: float      # (기준 AUC - 새 표본 AUC)의 95% 구간
    high: float

    @property
    def drop(self) -> float:
        return self.auc_ref - self.auc_cur

    @property
    def status(self) -> str:
        if self.low <= 0:           # 구간이 0을 포함하면 떨어졌다고 말할 수 없다
            return "통과"
        return "실패" if self.drop >= DROP_FAIL else "주의" if self.drop >= DROP_WARN else "관찰"


def check(y_ref, s_ref, y_cur, s_cur, n_boot: int = 300, seed: int = 0) -> Performance:
    """두 표본을 각각 다시 뽑아 AUC 차이의 구간을 낸다."""
    y_ref, s_ref, y_cur, s_cur = (np.asarray(a) for a in (y_ref, s_ref, y_cur, s_cur))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i, j = rng.integers(0, len(y_ref), len(y_ref)), rng.integers(0, len(y_cur), len(y_cur))
        if len(set(y_ref[i])) < 2 or len(set(y_cur[j])) < 2:
            continue
        diffs.append(roc_auc_score(y_ref[i], s_ref[i]) - roc_auc_score(y_cur[j], s_cur[j]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return Performance(float(roc_auc_score(y_ref, s_ref)), float(roc_auc_score(y_cur, s_cur)), float(low), float(high))
