"""공개 자료를 읽는다. 자료 파일은 저장소에 넣지 않는다."""
from __future__ import annotations

import io

import pandas as pd

GERMAN_COLUMNS = ["status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration",
                  "installment_rate", "personal_status_sex", "other_debtors", "present_residence", "property", "age",
                  "other_installment_plans", "housing", "number_credits", "job", "people_liable", "telephone",
                  "foreign_worker", "credit_risk"]
HOUSING = {"A151": "rent", "A152": "own", "A153": "for free"}


def parse_german(text: str) -> pd.DataFrame:
    """UCI Statlog German Credit의 german.data(공백 구분, 머리글 없음)를 표로 바꾼다.
    정답은 원본에서 1 = 우량, 2 = 부실이다. 여기서는 1 = 우량, 0 = 부실로 바꾼다. 주거 형태만 말로 풀고 나머지 부호는 그대로 둔다."""
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=GERMAN_COLUMNS)
    if set(df["credit_risk"].unique()) - {1, 2}:
        raise ValueError("정답 열에 1, 2가 아닌 값이 있다")
    df["credit_risk"] = (df["credit_risk"] == 1).astype(int)
    df["housing"] = df["housing"].map(HOUSING).fillna(df["housing"])
    return df
