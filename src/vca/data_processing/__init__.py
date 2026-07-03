"""Data acquisition, canonical schema, and cleaning for the virtual control arm."""

from vca.data_processing.schema import (
    BASELINE_REQUIRED,
    EVENTS_REQUIRED,
    LONGITUDINAL_REQUIRED,
    TrialData,
    validate_trial_data,
)

__all__ = [
    "TrialData",
    "validate_trial_data",
    "BASELINE_REQUIRED",
    "LONGITUDINAL_REQUIRED",
    "EVENTS_REQUIRED",
]
