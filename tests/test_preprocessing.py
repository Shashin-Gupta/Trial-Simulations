import numpy as np

from vca.models.preprocessing import CovariatePreprocessor


def test_fit_transform_shape_and_finite(synthetic_td):
    pre = CovariatePreprocessor()
    X = pre.fit_transform(synthetic_td.baseline)
    assert X.shape[0] == synthetic_td.n_patients
    assert X.shape[1] == pre.n_features
    assert np.isfinite(X).all()


def test_handles_missing_and_adds_pdl1_indicator(synthetic_td):
    pre = CovariatePreprocessor().fit(synthetic_td.baseline)
    # pdl1_tps is ~60% missing in the generator -> a missing-indicator feature.
    names = " ".join(pre.feature_names_)
    assert "pdl1" in names
    assert any("missing" in n or "indicator" in n for n in pre.feature_names_)


def test_transform_unseen_categories(synthetic_td):
    pre = CovariatePreprocessor().fit(synthetic_td.baseline)
    df = synthetic_td.baseline.head(3).copy()
    df.loc[df.index[0], "histology"] = "brand_new_category"
    X = pre.transform(df)  # handle_unknown='ignore' -> no crash
    assert X.shape == (3, pre.n_features)
    assert np.isfinite(X).all()


def test_only_present_columns_used():
    import pandas as pd

    df = pd.DataFrame(
        {"patient_id": ["a", "b"], "age": [60, 70], "sex": ["M", "F"]}
    )
    pre = CovariatePreprocessor().fit(df)
    assert pre.numeric_ == ["age"]
    assert pre.categorical_ == ["sex"]
