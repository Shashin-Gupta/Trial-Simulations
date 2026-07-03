"""Virtual Control Arms — generative patient-level simulation for oncology trials.

A research toolkit that fits a generative model to historical single-arm /
comparator-arm oncology data and simulates realistic patient-level trajectories
(RECIST tumour size over time, progression-free survival, overall survival) for
a *virtual control arm*. The initial, validated indication is advanced
non-small-cell lung cancer (NSCLC); see ``docs/methodology.md``.

Package layout
--------------
- ``vca.data_processing`` : canonical schema, synthetic data, dataset loaders.
- ``vca.models``          : the ``TrajectoryModel`` interface and implementations.
- ``vca.validation``      : held-out validation metrics and orchestration.
- ``vca.viz``             : plotting helpers.
- ``vca.product``         : Phase 3 product wrapper (stubbed).
"""

__version__ = "0.1.0"

from vca.data_processing.schema import TrialData  # noqa: E402,F401

__all__ = ["TrialData", "__version__"]
