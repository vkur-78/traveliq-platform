"""
TRAVELIQ — Hybrid Flight + Hotel Travel Package Recommendation Engine.

Productionizes the exact multi-dataset recommendation methodology developed and
validated in Phase 1 (notebooks/03_travel_recommendation.ipynb):
  - Joins hotels.csv, round-trip flights.csv, and users.csv via travelCode & userCode.
  - Explicitly surfaces the 1-hotel-per-city dataset structure (9 cities <-> 9 hotels).
  - Combines User-Based Collaborative Filtering (top-30 cosine neighbors) with
    transparent multi-factor scoring (Budget Fit 35%, Duration Match 25%,
    Personalization / CF Affinity 25%, Popularity & Reliability 15%).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import DATASET_DIR

ONE_HOTEL_LIMITATION_NOTE = (
    "Dataset Transparency: Each of the 9 Brazilian cities in hotels.csv contains "
    "exactly one partner hotel with one fixed daily rate. TRAVELIQ joins round-trip "
    "flights.csv, hotels.csv, and users.csv to rank 1,568 real Flight + Hotel packages "
    "across destinations, cabin classes, agencies, and trip durations."
)


class TravelPackageRecommender:
    """Hybrid collaborative + content/budget travel package recommender."""

    def __init__(
        self,
        flights_df: pd.DataFrame,
        hotels_df: pd.DataFrame,
        users_df: pd.DataFrame,
    ) -> None:
        self.flights = flights_df.copy()
        self.hotels = hotels_df.copy()
        self.users = users_df.copy()

        if not pd.api.types.is_datetime64_any_dtype(self.flights["date"]):
            self.flights["date"] = pd.to_datetime(self.flights["date"], format="%m/%d/%Y")
        if not pd.api.types.is_datetime64_any_dtype(self.hotels["date"]):
            self.hotels["date"] = pd.to_datetime(self.hotels["date"], format="%m/%d/%Y")

        self._build_indices()

    @classmethod
    def from_csv_dir(cls, data_dir: Optional[Path] = None) -> "TravelPackageRecommender":
        """Load flights.csv, hotels.csv, and users.csv from disk and initialize engine."""
        base = Path(data_dir) if data_dir is not None else DATASET_DIR
        flights_df = pd.read_csv(base / "flights.csv")
        hotels_df = pd.read_csv(base / "hotels.csv")
        users_df = pd.read_csv(base / "users.csv")
        return cls(flights_df=flights_df, hotels_df=hotels_df, users_df=users_df)

    def _build_indices(self) -> None:
        """Build unified Flight+Hotel catalog, CF affinity matrix, and user profiles."""
        # 1. Round-trip flight trips (2 legs per travelCode)
        flights_sorted = self.flights.sort_values(["travelCode", "date"]).reset_index(drop=True)
        flight_trips = flights_sorted.groupby("travelCode", as_index=False).agg(
            origin=("from", "first"),
            flight_dest=("to", "first"),
            flightType=("flightType", "first"),
            agency=("agency", "first"),
            flight_legs=("price", "count"),
            flight_roundtrip_price=("price", "sum"),
            avg_leg_time=("time", "mean"),
            leg_distance=("distance", "first"),
            departure_date=("date", "first"),
        )

        # 2. Join hotels + round-trip flights + users (avoid column name collision)
        self.trips = (
            self.hotels.rename(columns={"name": "hotel_name"})
            .merge(flight_trips, on="travelCode", how="inner")
            .merge(
                self.users.rename(columns={"code": "userCode", "name": "user_name"}),
                on="userCode",
                how="left",
            )
        )
        self.trips["package_total"] = self.trips["total"] + self.trips["flight_roundtrip_price"]

        # 3. Distinct Flight + Hotel package catalog (1,568 packages)
        self.package_catalog = (
            self.trips.groupby(
                ["origin", "place", "hotel_name", "flightType", "agency", "days"],
                as_index=False,
            )
            .agg(
                hotel_price_per_day=("price", "first"),
                hotel_total=("total", "first"),
                flight_roundtrip_price=("flight_roundtrip_price", "mean"),
                package_total=("package_total", "mean"),
                avg_leg_time=("avg_leg_time", "mean"),
                leg_distance=("leg_distance", "first"),
                historical_bookings=("travelCode", "count"),
                unique_travelers=("userCode", "nunique"),
            )
            .rename(columns={"place": "destination"})
        )

        pkg_log_pop = np.log1p(self.package_catalog["historical_bookings"])
        denom_pop = max(float(pkg_log_pop.max() - pkg_log_pop.min()), 1e-9)
        self.package_catalog["popularity_score"] = (pkg_log_pop - pkg_log_pop.min()) / denom_pop

        # 4. City -> single hotel lookup & summary
        self.city_hotel_summary = (
            self.hotels.groupby(["place", "name"], as_index=False)
            .agg(
                bookings=("travelCode", "count"),
                unique_users=("userCode", "nunique"),
                price_per_day=("price", "first"),
                avg_days=("days", "mean"),
                avg_hotel_total=("total", "mean"),
            )
            .rename(columns={"place": "destination", "name": "hotel_name"})
            .sort_values("bookings", ascending=False)
            .reset_index(drop=True)
        )
        self.city_to_hotel: Dict[str, str] = dict(
            zip(self.city_hotel_summary["destination"], self.city_hotel_summary["hotel_name"])
        )

        # 5. User-Based Collaborative Filtering affinity (matching Notebook 03)
        self.all_hotels: List[str] = sorted(self.hotels["name"].unique().tolist())
        uh_mat = (
            self.hotels.groupby(["userCode", "name"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=self.all_hotels, fill_value=0)
        )
        uh_norm = uh_mat.div(uh_mat.sum(axis=1), axis=0).fillna(0.0)
        pop_series = self.hotels["name"].value_counts(normalize=True).reindex(
            self.all_hotels, fill_value=0.0
        )

        sim_mat = cosine_similarity(uh_norm.values)
        np.fill_diagonal(sim_mat, 0.0)

        cf_raw = np.zeros_like(uh_norm.values)
        top_k_nbrs = min(30, max(len(uh_norm) - 1, 1))
        for i in range(len(uh_norm)):
            nbr_idx = np.argsort(sim_mat[i])[-top_k_nbrs:]
            weights = sim_mat[i, nbr_idx]
            w_sum = float(weights.sum())
            if w_sum > 0:
                nbr_pref = (weights[:, None] * uh_norm.values[nbr_idx]).sum(axis=0) / w_sum
                cf_raw[i] = 0.5 * uh_norm.values[i] + 0.5 * nbr_pref
            else:
                cf_raw[i] = pop_series.values

        cf_df = pd.DataFrame(cf_raw, index=uh_norm.index, columns=self.all_hotels)
        r_min = cf_df.min(axis=1)
        r_max = cf_df.max(axis=1)
        self.cf_affinity_norm_df = cf_df.sub(r_min, axis=0).div(
            (r_max - r_min).replace(0, 1), axis=0
        )

        p_span = max(float(pop_series.max() - pop_series.min()), 1e-9)
        self.global_hotel_pop_norm: Dict[str, float] = (
            (pop_series - pop_series.min()) / p_span
        ).to_dict()

        # 6. User travel profiles
        self.user_profiles = (
            self.trips.groupby("userCode")
            .agg(
                total_trips=("travelCode", "count"),
                dominant_origin=("origin", lambda s: s.value_counts().index[0]),
                preferred_flight_type=("flightType", lambda s: s.value_counts().index[0]),
                avg_package_spend=("package_total", "mean"),
                favorite_destination=("place", lambda s: s.value_counts().index[0]),
            )
            .reset_index()
            .merge(self.users.rename(columns={"code": "userCode"}), on="userCode", how="left")
            .set_index("userCode")
        )

        self.valid_cities: List[str] = sorted(
            set(self.package_catalog["destination"]).union(set(self.package_catalog["origin"]))
        )
        self.valid_flight_types: List[str] = sorted(
            self.package_catalog["flightType"].unique().tolist()
        )
        self.known_user_codes: List[int] = sorted(int(u) for u in self.user_profiles.index)

    def get_user_profile(self, user_code: int) -> Optional[Dict[str, Any]]:
        """Return summary dictionary for a historical traveler, or None if unknown."""
        if user_code not in self.user_profiles.index:
            return None
        row = self.user_profiles.loc[user_code]
        return {
            "user_code": int(user_code),
            "name": str(row.get("name", f"User #{user_code}")),
            "company": str(row.get("company", "Unknown")),
            "age": int(row["age"]) if pd.notna(row.get("age")) else None,
            "total_trips": int(row["total_trips"]),
            "dominant_origin": str(row["dominant_origin"]),
            "preferred_flight_type": str(row["preferred_flight_type"]),
            "favorite_destination": str(row["favorite_destination"]),
            "avg_package_spend": round(float(row["avg_package_spend"]), 2),
        }

    @staticmethod
    def _build_reason_text(
        row: pd.Series,
        requested_days: int,
        max_budget: float,
        is_personalized: bool,
        user_pref_class: Optional[str],
    ) -> str:
        """Generate a human-readable explanation of why a package ranked highly."""
        reasons: List[str] = []
        savings = max_budget - float(row["package_total"])
        savings_pct = (savings / max_budget * 100.0) if max_budget > 0 else 0.0

        if int(row["days"]) == requested_days:
            reasons.append(f"exact {requested_days}-day stay match")
        else:
            reasons.append(f"close {int(row['days'])}-day stay alternative")

        if savings_pct >= 25:
            reasons.append(f"strong budget value (${savings:,.0f} under budget)")
        elif savings >= 0:
            reasons.append(f"fits within budget (${savings:,.0f} remaining)")

        if is_personalized:
            if float(row["cf_hotel_affinity"]) >= 0.65:
                reasons.append("high collaborative-filtering destination affinity")
            if user_pref_class and row["flightType"] == user_pref_class:
                reasons.append(f"matches preferred {user_pref_class} cabin")
        elif float(row["popularity_score"]) >= 0.65:
            reasons.append(f"popular route ({int(row['historical_bookings'])} past trips)")

        return "; ".join(reasons).capitalize() + "."

    def recommend(
        self,
        days: int,
        max_budget: float,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        flight_type: Optional[str] = None,
        user_code: Optional[int] = None,
        top_n: int = 5,
        strict_days: bool = False,
        diverse_destinations: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Rank Flight + Hotel packages matching budget, duration, route, and personalization.

        Returns
        -------
        (recommendations_df, metadata_dict)
        """
        notes: List[str] = []

        # 1. Input validation
        if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or int(days) <= 0:
            return pd.DataFrame(), {
                "status": "error",
                "message": "days must be a positive integer (1 to 4 recommended).",
            }
        days = int(days)

        if (
            max_budget is None
            or isinstance(max_budget, bool)
            or not isinstance(max_budget, (int, float, np.number))
            or not np.isfinite(max_budget)
            or float(max_budget) <= 0
        ):
            return pd.DataFrame(), {
                "status": "error",
                "message": "max_budget must be a finite number greater than 0.",
            }
        max_budget = float(max_budget)

        if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or int(top_n) <= 0:
            return pd.DataFrame(), {
                "status": "error",
                "message": "top_n must be a positive integer.",
            }
        top_n = int(top_n)

        if origin is not None and origin not in self.valid_cities:
            return pd.DataFrame(), {
                "status": "error",
                "message": f"Unknown origin city '{origin}'. Valid cities: {self.valid_cities}",
            }
        if destination is not None and destination not in self.valid_cities:
            return pd.DataFrame(), {
                "status": "error",
                "message": f"Unknown destination city '{destination}'. Valid cities: {self.valid_cities}",
            }
        if origin is not None and destination is not None and origin == destination:
            return pd.DataFrame(), {
                "status": "error",
                "message": "origin and destination cannot be the same city.",
            }
        if flight_type is not None and flight_type not in self.valid_flight_types:
            return pd.DataFrame(), {
                "status": "error",
                "message": (
                    f"Unknown flight_type '{flight_type}'. "
                    f"Valid flight types: {self.valid_flight_types}"
                ),
            }

        # 2. User personalization profile or cold-start fallback
        is_personalized = False
        user_pref_class: Optional[str] = None
        inferred_origin = origin

        if user_code is not None:
            if user_code in self.user_profiles.index and user_code in self.cf_affinity_norm_df.index:
                is_personalized = True
                profile = self.user_profiles.loc[user_code]
                user_pref_class = str(profile["preferred_flight_type"])
                hotel_affinity_map = self.cf_affinity_norm_df.loc[user_code].to_dict()
                if inferred_origin is None and destination != profile["dominant_origin"]:
                    inferred_origin = str(profile["dominant_origin"])
                    notes.append(
                        f"Inferred Traveler #{user_code}'s typical departure hub: {inferred_origin}."
                    )
            else:
                hotel_affinity_map = self.global_hotel_pop_norm
                notes.append(
                    f"Traveler #{user_code} has no booking history (cold-start fallback applied)."
                )
        else:
            hotel_affinity_map = self.global_hotel_pop_norm
            notes.append("Anonymous mode: ranked by budget fit, duration match, and route popularity.")

        # 3. Candidate filtering
        candidates = self.package_catalog[self.package_catalog["package_total"] <= max_budget].copy()

        if inferred_origin is not None:
            candidates = candidates[candidates["origin"] == inferred_origin]
        if destination is not None:
            city_hotel = self.city_to_hotel.get(destination, "Partner Hotel")
            notes.append(
                f"Single-hotel city notice: {destination} has 1 hotel ({city_hotel}). "
                "Showing ranked round-trip Flight + Hotel packages for this destination."
            )
            candidates = candidates[candidates["destination"] == destination]
        if flight_type is not None:
            candidates = candidates[candidates["flightType"] == flight_type]
        if strict_days:
            candidates = candidates[candidates["days"] == days]

        if candidates.empty:
            return pd.DataFrame(), {
                "status": "no_match",
                "message": (
                    f"No Flight + Hotel packages found within ${max_budget:,.2f}. "
                    "Try increasing your budget, relaxing strict duration matching, "
                    "or clearing the destination/cabin filter."
                ),
                "notes": notes,
                "dataset_limitation": ONE_HOTEL_LIMITATION_NOTE,
            }

        # 4. Transparent Multi-Factor Scoring (0 - 100)
        candidates["budget_savings"] = max_budget - candidates["package_total"]
        candidates["budget_savings_pct"] = candidates["budget_savings"] / max_budget
        candidates["budget_score"] = (0.5 + 0.5 * candidates["budget_savings_pct"]).clip(0.0, 1.0)

        candidates["days_diff"] = (candidates["days"] - days).abs()
        candidates["duration_score"] = (1.0 - 0.35 * candidates["days_diff"]).clip(0.0, 1.0)

        candidates["cf_hotel_affinity"] = candidates["hotel_name"].map(hotel_affinity_map).fillna(0.5)
        if user_pref_class is not None:
            class_bonus = (candidates["flightType"] == user_pref_class).astype(float)
            candidates["personal_score"] = 0.70 * candidates["cf_hotel_affinity"] + 0.30 * class_bonus
        else:
            candidates["personal_score"] = candidates["cf_hotel_affinity"]

        candidates["recommendation_score"] = 100.0 * (
            0.35 * candidates["budget_score"]
            + 0.25 * candidates["duration_score"]
            + 0.25 * candidates["personal_score"]
            + 0.15 * candidates["popularity_score"]
        )

        candidates = candidates.sort_values(
            by=["recommendation_score", "days_diff", "package_total"],
            ascending=[False, True, True],
        )

        # Surface diverse destinations first in exploration mode
        if destination is None and diverse_destinations:
            top_diverse = candidates.drop_duplicates(subset=["destination"], keep="first")
            if len(top_diverse) >= top_n:
                selected = top_diverse.head(top_n).copy()
            else:
                remaining = candidates.drop(index=top_diverse.index).head(top_n - len(top_diverse))
                selected = pd.concat([top_diverse, remaining]).head(top_n).copy()
        else:
            selected = candidates.head(top_n).copy()

        selected = selected.reset_index(drop=True)
        selected.insert(0, "rank", range(1, len(selected) + 1))
        selected["recommendation_reason"] = selected.apply(
            lambda r: self._build_reason_text(
                r,
                requested_days=days,
                max_budget=max_budget,
                is_personalized=is_personalized,
                user_pref_class=user_pref_class,
            ),
            axis=1,
        )

        output_cols = [
            "rank",
            "origin",
            "destination",
            "hotel_name",
            "days",
            "flightType",
            "agency",
            "hotel_price_per_day",
            "flight_roundtrip_price",
            "hotel_total",
            "package_total",
            "budget_savings",
            "avg_leg_time",
            "leg_distance",
            "historical_bookings",
            "budget_score",
            "duration_score",
            "personal_score",
            "popularity_score",
            "recommendation_score",
            "recommendation_reason",
        ]

        return selected[output_cols], {
            "status": "ok",
            "personalized": is_personalized,
            "origin_used": inferred_origin,
            "destination_used": destination,
            "total_eligible_packages": int(len(candidates)),
            "returned_count": int(len(selected)),
            "notes": notes,
            "dataset_limitation": ONE_HOTEL_LIMITATION_NOTE,
        }
