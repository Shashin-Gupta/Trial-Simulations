import pytest

from vca.data_processing.schema import (
    SchemaError,
    TrialData,
    validate_trial_data,
)


def test_synthetic_passes_validation(synthetic_td):
    validate_trial_data(synthetic_td)  # should not raise


def test_required_columns_enforced(synthetic_td):
    bad = synthetic_td.baseline.drop(columns=["age"])
    td = TrialData(bad, synthetic_td.longitudinal, synthetic_td.events)
    with pytest.raises(SchemaError):
        td.validate()


def test_event_domain_enforced(synthetic_td):
    ev = synthetic_td.events.copy()
    ev.loc[0, "pfs_event"] = 2  # invalid
    td = TrialData(synthetic_td.baseline, synthetic_td.longitudinal, ev)
    with pytest.raises(SchemaError):
        td.validate()


def test_referential_integrity(synthetic_td):
    ev = synthetic_td.events.copy()
    ev.loc[0, "patient_id"] = "GHOST"
    td = TrialData(synthetic_td.baseline, synthetic_td.longitudinal, ev)
    with pytest.raises(SchemaError):
        td.validate()


def test_split_is_on_patients_not_rows(synthetic_td):
    train, test = synthetic_td.train_test_split(0.3, seed=1)
    # Disjoint patients.
    assert set(train.patient_ids).isdisjoint(set(test.patient_ids))
    # No patient's measurements are split across train and test.
    train_ids, test_ids = set(train.patient_ids), set(test.patient_ids)
    assert set(train.longitudinal.patient_id).issubset(train_ids)
    assert set(test.longitudinal.patient_id).issubset(test_ids)
    assert train.n_patients + test.n_patients == synthetic_td.n_patients


def test_covariates_exclude_identifiers(synthetic_td):
    cov = synthetic_td.covariates()
    for col in ("patient_id", "study_id", "treatment"):
        assert col not in cov.columns


def test_roundtrip_parquet(tmp_path, synthetic_td):
    synthetic_td.to_parquet(tmp_path, prefix="t")
    back = TrialData.from_parquet(tmp_path, prefix="t")
    assert back.n_patients == synthetic_td.n_patients
    assert set(back.baseline.columns) == set(synthetic_td.baseline.columns)
