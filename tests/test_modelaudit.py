"""아래 자료는 모두 지어낸 것이다."""
import numpy as np
import pandas as pd

from modelaudit import data, drift, fairness, perf, planted, report, robust, samples
from modelaudit.__main__ import main


def fake(n=600, seed=0):
    """German Credit과 열 이름만 같은 가짜 자료."""
    rng = np.random.default_rng(seed)
    age = rng.integers(19, 70, n)
    amount = rng.lognormal(8, 0.7, n).round()
    housing = rng.choice(["rent", "own", "for free"], n, p=[0.2, 0.7, 0.1])
    good = rng.random(n) < 1 / (1 + np.exp(-(1.2 - amount / 6000 + (housing == "own") * 0.6)))
    return pd.DataFrame({"age": age, "amount": amount, "housing": housing, "duration": rng.integers(6, 48, n),
                         "personal_status_sex": rng.choice(["a", "b"], n), "foreign_worker": rng.choice(["yes", "no"], n),
                         "credit_risk": good.astype(int)})


def test_psi_is_small_for_same_distribution_and_large_after_shift():
    rng = np.random.default_rng(1)
    a, b = pd.Series(rng.normal(0, 1, 2000)), pd.Series(rng.normal(0, 1, 2000))
    assert drift.psi(a, b).psi < 0.03
    assert drift.psi(a, b + 1).psi > 0.25
    assert drift.psi(pd.Series(list("aabbb") * 100), pd.Series(list("abbbb") * 100)).psi > 0.1


def test_p_value_keeps_false_alarms_near_five_percent_when_nothing_changed():
    rng = np.random.default_rng(2)
    hits = [drift.psi(pd.Series(rng.normal(size=300)), pd.Series(rng.normal(size=120))).p_value < 0.05 for _ in range(600)]
    assert 0.02 < np.mean(hits) < 0.09


def test_small_sample_passes_convention_threshold_by_chance_but_not_the_adjusted_rule():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame(rng.normal(size=(500, 12)), columns=[f"x{i}" for i in range(12)])
    cur = pd.DataFrame(rng.normal(size=(40, 12)), columns=ref.columns)
    res = drift.check(ref, cur)
    assert any(r.status(adjusted=False) != "통과" for r in res)
    assert all(r.status() == "통과" for r in res)


def test_holm():
    assert np.allclose(drift.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_disparity_statuses():
    g = np.array(["a"] * 200 + ["b"] * 200)
    same = np.array(([True] * 140 + [False] * 60) * 2)
    worse = np.array([True] * 140 + [False] * 60 + [True] * 90 + [False] * 110)
    assert fairness.disparity(same, g).status == "통과"
    assert fairness.disparity(worse, g).status == "실패"
    tiny = fairness.disparity([True, True, True, False] * 3 + [True, False] * 2, ["a"] * 12 + ["b"] * 4)
    assert tiny.status == "주의"    # 비는 낮지만 표본이 작아 유의하지 않다


def test_opportunity_looks_only_at_truly_good_applicants():
    g = np.array(["a"] * 100 + ["b"] * 100)
    y = np.array([1] * 50 + [0] * 50 + [1] * 50 + [0] * 50)
    approved = np.array([True] * 50 + [False] * 50 + [True] * 50 + [True] * 50)
    assert fairness.opportunity(approved, y, g).ratio == 1.0
    assert fairness.disparity(approved, g).ratio == 0.5


def test_flip_is_zero_when_model_ignores_the_column_and_catches_a_penalty():
    df = fake()
    X, y = df.drop(columns=["credit_risk"]), df["credit_risk"]
    model = samples.train(X, y, exclude=("age", "personal_status_sex", "foreign_worker"))
    cut = samples.cutoff_for(model(X))
    assert fairness.flip(model, X, "age", 22, 40, cut).status == "통과"
    punished = samples.with_penalty(model, lambda t: t["age"] <= 25, 0.15)
    assert fairness.flip(punished, X, "age", 22, 40, cut).status == "실패"


def test_planted_changes_do_what_they_say():
    df, rng = fake(), np.random.default_rng(0)
    member = df["age"] <= 25
    y2 = samples.biased_labels(df["credit_risk"], member, 0.5, rng)
    assert (y2[~member] == df["credit_risk"][~member]).all() and y2[member].sum() < df["credit_risk"][member].sum()
    assert (samples.remixed(df, "housing", "rent", 4, rng)["housing"] == "rent").mean() > (df["housing"] == "rent").mean() + 0.1
    assert np.allclose(samples.scaled(df, "amount", 1.3)["amount"], df["amount"] * 1.3)


def test_parse_official_german_format():
    text = ("A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1\n"
            "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A151 1 A173 1 A191 A201 2\n")
    df = data.parse_german(text)
    assert df["credit_risk"].tolist() == [1, 0] and df["housing"].tolist() == ["own", "rent"] and df["amount"].tolist() == [1169, 5951]


def test_report_and_commands_run_end_to_end(tmp_path):
    fake(500).to_csv(tmp_path / "d.csv", index=False)
    assert main(["report", "--data", str(tmp_path / "d.csv"), "--out", str(tmp_path)]) == 0
    assert main(["planted", "--data", str(tmp_path / "d.csv"), "--out", str(tmp_path), "--seeds", "2"]) == 0
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "신뢰성" in text and "바꿔 넣기" in text and "고위험" in text and "설명가능성 | 검사 안 함" in text
    assert "간접 차별" in (tmp_path / "planted.md").read_text(encoding="utf-8")
    assert "불안정" in (tmp_path / "planted.md").read_text(encoding="utf-8") and "### 강건성" in text
    assert report.overall(["통과", "실패", "주의"]) == "실패"


def test_significant_but_small_drift_is_reported_as_observation_not_pass():
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 60000)})
    cur = pd.DataFrame({"x": rng.normal(0.12, 1, 60000)})
    r = drift.check(ref, cur)[0]
    assert r.psi < 0.10 and r.status(adjusted=False) == "통과" and r.status() == "관찰"
    assert drift.check(ref, pd.DataFrame({"x": rng.normal(0, 1, 60000)}))[0].status() == "통과"


def test_versus_isolates_what_the_audited_model_adds():
    rng = np.random.default_rng(6)
    n = 8000
    member = rng.random(n) < 0.2
    bench = rng.random(n) < np.where(member, 0.5, 0.7)        # 자료에 원래 있던 차이
    same = bench.copy()
    harsher = bench & ~(member & (rng.random(n) < 0.3))       # 집단에서만 승인의 30%를 더 거절
    assert fairness.versus(same, bench, member).status == "통과"
    v = fairness.versus(harsher, bench, member)
    assert v.status == "실패" and v.extra < -0.1 and abs(v.out_group) < 1e-12


def test_benchmark_model_runs_with_missing_values():
    df = fake(400)
    df.loc[::7, "amount"] = np.nan
    X, y = df.drop(columns=["credit_risk"]), df["credit_risk"]
    assert samples.train_benchmark(X, y, exclude=("age",))(X).shape == (400,)
    assert samples.train(X, y, exclude=("age",))(X).shape == (400,)


def test_risk_grade_follows_published_bands_and_high_impact_rule():
    assert report.grade(10, False)[0] == "저위험"
    assert report.grade(25, False)[0] == "중위험"
    assert report.grade(50, False)[0] == "고위험"
    assert report.grade(80, False)[0] == "초고위험"
    assert report.grade(10, True)[0] == "고위험"        # 고영향 AI는 점수가 낮아도 고위험
    assert report.grade(80, True)[0] == "초고위험"
    assert report.grade(None, True)[0] == "고위험"
    assert report.grade(None, False)[0] == "산정 안 함"


def test_unchecked_area_is_shown_and_never_counted_as_pass():
    assert report.overall(["검사 안 함", "통과"]) == "통과"
    assert report.overall(["검사 안 함"]) == "검사 안 함"
    text = report.build("x", [], [])
    assert "검사하지 않은 영역: 모델 성능 관리, 데이터 품질, 공정성·편향성, 설명가능성" in text


def test_performance_drop_is_flagged_only_when_real():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 4000)
    good = y + rng.normal(0, 0.8, 4000)
    assert perf.check(y[:2000], good[:2000], y[2000:], good[2000:], n_boot=100).status == "통과"
    worse = y[2000:] + rng.normal(0, 2.5, 2000)
    assert perf.check(y[:2000], good[:2000], y[2000:], worse, n_boot=100).status == "실패"


def test_ripple_model_flips_on_one_percent_change_and_smooth_model_does_not():
    df = fake(1500)
    X, y = df.drop(columns=["credit_risk"]), df["credit_risk"]
    model = samples.train(X, y, exclude=("age", "personal_status_sex", "foreign_worker"))
    cut = samples.cutoff_for(model(X))
    assert all(r.status == "통과" for r in robust.check(model, X, cut))
    shaky = samples.with_ripple(model, "amount", float(np.std(model(X))))
    res = {r.column: r for r in robust.check(shaky, X, samples.cutoff_for(shaky(X)))}
    assert res["amount"].status == "실패" and res["duration"].status == "통과"
