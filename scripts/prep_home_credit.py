"""Kaggle 'Home Credit Default Risk'의 application_train.csv를 점검용 표로 줄인다.

    python scripts/prep_home_credit.py 받은폴더/application_train.csv

원본의 TARGET은 1 = 연체다. 여기서는 credit_risk 1 = 우량으로 뒤집는다. 나이는 DAYS_BIRTH에서 햇수로 바꾼다.
신청 날짜는 이 파일에 없다. 그래서 실제 시점 변화는 이 자료로 볼 수 없다.
"""
import sys
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modelaudit import provenance  # noqa: E402

KEEP = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_CHILDREN", "REGION_RATING_CLIENT",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
        "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]

if __name__ == "__main__":
    raw = pd.read_csv(sys.argv[1])
    raw = raw[raw["CODE_GENDER"].isin(["F", "M"])]          # 성별 미상 4건 제외
    out = raw[KEEP].copy()
    out["years_employed"] = np.where(raw["DAYS_EMPLOYED"] > 0, np.nan, -raw["DAYS_EMPLOYED"] / 365.25)   # 양수(365243)는 무직·연금 표시
    out["age"] = (-raw["DAYS_BIRTH"] / 365.25).round(1)
    out["gender"] = raw["CODE_GENDER"]
    out["credit_risk"] = 1 - raw["TARGET"]
    path = Path(__file__).resolve().parents[1] / "data" / "home_credit.csv"
    out.to_csv(path, index=False)
    raw_sha = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
    provenance.write(path, source="Kaggle Home Credit Default Risk (application_train.csv)",
                     url="https://www.kaggle.com/competitions/home-credit-default-risk/data", rows=len(out), raw_sha256=raw_sha,
                     steps="성별 미상 제외, 20개 열만 남김, 근속·나이를 햇수로, TARGET 뒤집기")
    print(f"{len(out)}행, 우량 {int(out.credit_risk.sum())}건 -> {path}")
