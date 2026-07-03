import numpy as np

from vca.data_processing.synthetic import make_synthetic_nsclc, tgi_trajectory


def test_shapes_and_schema():
    td = make_synthetic_nsclc(150, seed=5)
    assert td.n_patients == 150
    assert len(td.events) == 150
    assert (td.longitudinal.groupby("patient_id").size() >= 1).all()
    td.validate()


def test_determinism():
    a = make_synthetic_nsclc(100, seed=7)
    b = make_synthetic_nsclc(100, seed=7)
    assert a.events.equals(b.events)
    assert a.longitudinal.equals(b.longitudinal)


def test_pfs_leq_os():
    td = make_synthetic_nsclc(300, seed=9)
    e = td.events
    assert (e.pfs_time_days <= e.os_time_days + 1e-6).all()


def test_return_truth_recoverable_coupling():
    td, truth = make_synthetic_nsclc(300, seed=11, return_truth=True)
    # Faster growth should be associated with shorter PFS (negative correlation).
    e = td.events.set_index("patient_id").loc[truth.patient_id]
    r = np.corrcoef(truth.growth_rate, e.pfs_time_days.to_numpy())[0, 1]
    assert r < 0


def test_tgi_trajectory_starts_at_baseline():
    y = tgi_trajectory(60.0, 0.004, 0.002, np.array([0.0, 100.0]))
    assert np.isclose(y[0], 60.0)
    assert (y > 0).all()
