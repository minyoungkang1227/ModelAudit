"""검증 대상으로 세워 두는 표본 모델과, 일부러 심는 문제.

심은 문제는 크기를 알고 있으므로 '몇 개 중 몇 개를 잡았는가'를 셀 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .fairness import Model


def train(df: pd.DataFrame, y, exclude: tuple[str, ...] = ()) -> Model:
    """로지스틱 회귀 표본 모델. exclude에 적은 열(보호 속성)은 모델에 넣지 않는다."""
    cols = [c for c in df.columns if c not in exclude]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    pipe = make_pipeline(
        ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), cat)]),
        LogisticRegression(max_iter=2000),
    )
    pipe.fit(df[cols], np.asarray(y))
    return lambda X: pipe.predict_proba(X[cols])[:, 1]


def cutoff_for(scores, approve_share: float = 0.7) -> float:
    """승인 비율이 approve_share가 되는 기준 점수."""
    return float(np.quantile(scores, 1 - approve_share))


# ---- 심는 문제: 공정성 ----

def with_penalty(model: Model, member, delta: float) -> Model:
    """직접 차별: member(X)가 참인 사람의 점수를 delta만큼 깎는다. 모델이 집단 값을 직접 쓰는 경우다."""
    return lambda X: model(X) - delta * np.asarray(member(X), float)


def biased_labels(y, member, share: float, rng) -> np.ndarray:
    """간접 차별: 집단에 속한 우량 고객의 일부를 학습용 정답에서 부실로 바꾼다.
    이 정답으로 배운 모델은 집단 값을 입력으로 받지 않아도 그 집단과 닮은 다른 변수로 깎는다."""
    y = np.asarray(y).copy()
    idx = np.flatnonzero((y == 1) & np.asarray(member, bool))
    y[rng.choice(idx, int(round(share * len(idx))), replace=False)] = 0
    return y


# ---- 심는 문제: 드리프트 ----

def scaled(df: pd.DataFrame, column: str, factor: float) -> pd.DataFrame:
    """숫자 변수 하나를 factor배 한다(예: 대출 금액이 전반적으로 커짐)."""
    return df.assign(**{column: df[column] * factor})


def remixed(df: pd.DataFrame, column: str, value, weight: float, rng) -> pd.DataFrame:
    """column == value인 사람이 weight배 더 자주 뽑히도록 다시 뽑는다(고객 구성이 바뀜)."""
    w = np.where(df[column] == value, weight, 1.0)
    return df.iloc[rng.choice(len(df), len(df), replace=True, p=w / w.sum())].reset_index(drop=True)
