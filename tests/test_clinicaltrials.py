"""Offline parsing tests + one opt-in live test for the ClinicalTrials.gov client."""

import pytest

from vca.data_processing import clinicaltrials as ct


def _fake_study():
    return {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": 400}},
        },
        "resultsSection": {
            "outcomeMeasuresModule": {
                "outcomeMeasures": [
                    {
                        "type": "PRIMARY",
                        "title": "Progression-free Survival (PFS)",
                        "paramType": "MEDIAN",
                        "unitOfMeasure": "Months",
                        "groups": [{"id": "OG000", "title": "Comparator"}],
                        "classes": [
                            {"categories": [
                                {"measurements": [{"groupId": "OG000", "value": "5.5"}]}
                            ]}
                        ],
                    },
                    {
                        "type": "SECONDARY",
                        "title": "Objective Response Rate",
                        "paramType": "NUMBER",
                        "unitOfMeasure": "percentage of participants",
                        "groups": [{"id": "OG000", "title": "Comparator"}],
                        "classes": [
                            {"categories": [
                                {"measurements": [{"groupId": "OG000", "value": "30"}]}
                            ]}
                        ],
                    },
                ]
            }
        },
    }


def test_parse_extracts_pfs_median_and_ignores_non_time():
    rows = ct.parse_outcome_medians(_fake_study())
    assert len(rows) == 1  # ORR percentage dropped
    r = rows[0]
    assert r["endpoint"] == "PFS"
    assert r["arm"] == "Comparator"
    assert r["median_days"] == pytest.approx(5.5 * ct.MONTHS_TO_DAYS, abs=0.1)  # module rounds to 1dp


def test_classify_endpoint():
    assert ct._classify_endpoint("Overall Survival (OS)") == "OS"
    assert ct._classify_endpoint("Progression Free Survival") == "PFS"
    assert ct._classify_endpoint("Number of Adverse Events") is None


def test_to_days_units():
    assert ct._to_days(2, "Months") == pytest.approx(2 * 30.4375)
    assert ct._to_days(3, "Weeks") == pytest.approx(21)
    assert ct._to_days(10, "percentage of participants") is None


def test_build_table_summary():
    df = ct.build_benchmark_table([_fake_study(), _fake_study()])
    assert len(df) == 2
    assert set(df.endpoint) == {"PFS"}


@pytest.mark.live
def test_live_fetch_smoke():
    studies = ct.fetch_studies(max_studies=3)
    assert len(studies) >= 1
    assert "protocolSection" in studies[0]
