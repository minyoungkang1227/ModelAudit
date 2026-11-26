"""검증 대상으로 세워 두는 표본 모델과, 일부러 심는 문제.

심은 문제는 크기를 알고 있으므로 '몇 개 중 몇 개를 잡았는가'를 셀 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .fairness import Model


def train(df: pd.DataFrame, y, exclude: tuple[str, ...] = ()) -> Model:
    """로지스틱 회귀 표본 모델. exclude에 적은 열(보호 속성)은 모델에 넣지 않는다."""
    cols = [c for c in df.columns if c not in exclude]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    pipe = make_pipeline(
        ColumnTransformer([   # 빈 값은 숫자는 중앙값, 범주는 '없음'이라는 값으로 채운다
            ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="없음"), OneHotEncoder(handle_unknown="ignore")), cat),
        ]),
        LogisticRegression(max_iter=2000),
    )
    pipe.fit(df[cols], np.asarray(y))
    return lambda X: pipe.predict_proba(X[cols])[:, 1]


def train_explained(df: pd.DataFrame, y, exclude: tuple[str, ...] = ()):
    """표본 모델과 그 모델이 내놓는 거절 사유. 외부 공급자가 점수와 함께 사유를 주는 상황을 흉내 낸다.
    사유는 로지스틱 회귀의 변수별 기여(계수 x (값 - 평균))를 원래 열 단위로 합쳐, 점수를 가장 많이 깎은 열이다."""
    cols = [c for c in df.columns if c not in exclude]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    pre = ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("c", make_pipeline(SimpleImputer(strategy="constant", fill_value="없음"), OneHotEncoder(handle_unknown="ignore")), cat),
    ], sparse_threshold=0)
    Z = pre.fit_transform(df[cols])
    lr = LogisticRegression(max_iter=2000).fit(Z, np.asarray(y))
    sizes = [1] * len(num) + ([len(c) for c in pre.named_transformers_["c"].steps[-1][1].categories_] if cat else [])
    owner = np.repeat(np.arange(len(sizes)), sizes)      # 변환된 열이 원래 어느 열에서 왔는가
    names, mean, coef = np.array(num + cat, dtype=object), Z.mean(axis=0), lr.coef_[0]

    def model(X: pd.DataFrame) -> np.ndarray:
        return lr.predict_proba(pre.transform(X[cols]))[:, 1]

    def explainer(X: pd.DataFrame) -> np.ndarray:
        contrib = (pre.transform(X[cols]) - mean) * coef
        by_column = np.column_stack([contrib[:, owner == k].sum(axis=1) for k in range(len(sizes))])
        return names[by_column.argmin(axis=1)]

    return model, explainer


def lying(explainer, columns: list[str], share: float, seed: int = 0):
    """거짓 사유: 사람의 share만큼은 진짜 사유 대신 아무 변수나 말한다. 같은 사람에게는 늘 같은 거짓을 말한다."""
    def told(X: pd.DataFrame) -> np.ndarray:
        r = np.asarray(explainer(X), dtype=object).copy()
        rng = np.random.default_rng(seed)
        lie = rng.random(len(r)) < share
        r[lie] = rng.choice(np.array(columns, dtype=object), int(lie.sum()))
        return r
    return told


def train_benchmark(df: pd.DataFrame, y, exclude: tuple[str, ...] = ()) -> Model:
    """검증자가 따로 세우는 기준 모델. 표본 모델과 종류가 다른 것(부스팅)을 써서 같은 모델끼리 비교가 되지 않게 한다."""
    cols = [c for c in df.columns if c not in exclude]
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    enc = ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat)],
                            remainder="passthrough")
    pipe = make_pipeline(enc, HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0,
                                                             categorical_features=list(range(len(cat)))))
    pipe.fit(df[cols], np.asarray(y))
    return lambda X: pipe.predict_proba(X[cols])[:, 1]


def out_of_fold(df: pd.DataFrame, y, exclude: tuple[str, ...] = (), k: int = 5, seed: int = 0) -> np.ndarray:
    """기준 표본을 k조각으로 나눠, 각 조각의 점수를 그 조각을 빼고 배운 모델로 낸다.
    모델이 배운 자료에서 잰 구별력은 부풀려져 있다. 새 표본과 견줄 기준은 배우지 않은 자료에서 잰 값이어야 한다."""
    y = np.asarray(y)
    fold = np.random.default_rng(seed).permutation(len(df)) % k
    scores = np.empty(len(df))
    for i in range(k):
        scores[fold == i] = train(df[fold != i], y[fold != i], exclude)(df[fold == i])
    return scores


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


# ---- 심는 문제: 강건성 ----

def with_ripple(model: Model, column: str, amplitude: float, step: float = 0.02) -> Model:
    """불안정한 모델: column 값이 step(2%)만큼 변할 때마다 한 바퀴 도는 잔물결을 점수에 더한다.
    값이 1% 달라지면 점수가 최대 amplitude의 두 배까지 움직인다. 과하게 맞춘 모델이나 촘촘한 규칙이 만드는 모양이다."""
    def scored(X: pd.DataFrame) -> np.ndarray:
        x = np.log(np.clip(X[column].to_numpy(float), 1e-9, None))
        return model(X) + amplitude * np.sin(2 * np.pi * x / np.log1p(step))
    return scored


# ---- 심는 문제: 드리프트 ----

def scaled(df: pd.DataFrame, column: str, factor: float) -> pd.DataFrame:
    """숫자 변수 하나를 factor배 한다(예: 대출 금액이 전반적으로 커짐)."""
    return df.assign(**{column: df[column] * factor})


def remixed(df: pd.DataFrame, column: str, value, weight: float, rng) -> pd.DataFrame:
    """column == value인 사람이 weight배 더 자주 뽑히도록 다시 뽑는다(고객 구성이 바뀜)."""
    w = np.where(df[column] == value, weight, 1.0)
    return df.iloc[rng.choice(len(df), len(df), replace=True, p=w / w.sum())].reset_index(drop=True)
