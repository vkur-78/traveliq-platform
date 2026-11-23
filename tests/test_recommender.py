"""
Tests for TRAVELIQ Hybrid Travel & Hotel Package Recommender (app/recommender.py).
"""

import pytest

from app.recommender import ONE_HOTEL_LIMITATION_NOTE, TravelPackageRecommender


@pytest.fixture(scope="module")
def recommender():
    return TravelPackageRecommender.from_csv_dir()


def test_recommender_exposes_one_hotel_per_city_reality(recommender):
    assert len(recommender.city_to_hotel) == 9
    assert len(set(recommender.city_to_hotel.values())) == 9
    assert len(recommender.package_catalog) == 1568


def test_destination_explorer_returns_multiple_ranked_packages_within_budget(recommender):
    recs, meta = recommender.recommend(
        days=3,
        max_budget=2500.0,
        origin="Brasilia (DF)",
        user_code=0,
        top_n=5,
    )
    assert meta["status"] == "ok"
    assert meta["personalized"] is True
    assert len(recs) == 5
    assert (recs["package_total"] <= 2500.0).all()
    # Separate flight, hotel, and total costs present and consistent
    assert (
        (recs["flight_roundtrip_price"] + recs["hotel_total"] - recs["package_total"]).abs()
        < 0.05
    ).all()
    # Scores monotonically non-increasing and reasons populated
    assert recs["recommendation_score"].is_monotonic_decreasing
    assert recs["recommendation_reason"].str.len().gt(10).all()
    assert meta["dataset_limitation"] == ONE_HOTEL_LIMITATION_NOTE


def test_specific_destination_mode_explains_single_hotel_and_ranks_packages(recommender):
    recs, meta = recommender.recommend(
        days=4,
        max_budget=3200.0,
        origin="Brasilia (DF)",
        destination="Florianopolis (SC)",
        top_n=5,
    )
    assert meta["status"] == "ok"
    assert len(recs) == 5
    assert (recs["destination"] == "Florianopolis (SC)").all()
    assert (recs["hotel_name"] == "Hotel A").all()
    assert any("1 hotel" in note for note in meta["notes"])


def test_tight_budget_returns_empty_dataframe_and_no_match_status(recommender):
    recs, meta = recommender.recommend(
        days=3,
        max_budget=150.0,
        origin="Recife (PE)",
        top_n=5,
    )
    assert recs.empty
    assert meta["status"] == "no_match"
    assert "No Flight + Hotel packages found" in meta["message"]


def test_cold_start_unknown_user_falls_back_cleanly(recommender):
    recs, meta = recommender.recommend(
        days=2,
        max_budget=2200.0,
        origin="Sao Paulo (SP)",
        user_code=999999,
        top_n=3,
    )
    assert meta["status"] == "ok"
    assert meta["personalized"] is False
    assert len(recs) == 3
    assert any("cold-start" in note.lower() for note in meta["notes"])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(days=0, max_budget=2000.0),
        dict(days=-2, max_budget=2000.0),
        dict(days=2, max_budget=-500.0),
        dict(days=2, max_budget=0.0),
        dict(days=2, max_budget=2000.0, destination="Mumbai"),
        dict(days=2, max_budget=2000.0, origin="Recife (PE)", destination="Recife (PE)"),
        dict(days=2, max_budget=2000.0, flight_type="superLuxury"),
        dict(days=2, max_budget=2000.0, top_n=0),
    ],
)
def test_invalid_recommender_inputs_return_error_status(recommender, kwargs):
    recs, meta = recommender.recommend(**kwargs)
    assert recs.empty
    assert meta["status"] == "error"
    assert meta["message"]
