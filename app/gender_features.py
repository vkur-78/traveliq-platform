"""
TRAVELIQ — Shared Deterministic Feature Engineering for Gender Classification.

Matches the exact row-wise linguistic and demographic transformations trained
in notebooks/02_gender_classification.ipynb (Phase 1).
"""

import pandas as pd

EXPERIMENTAL_GENDER_NOTICE = (
    "Experimental model demonstration only: predictions are statistical estimates "
    "derived from historical first-name patterns and demographic fields. They are NOT "
    "verified personal identity attributes and must NOT be used for consequential, "
    "automated, or policy-sensitive decisions."
)


def build_gender_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic row-wise features expected by the Phase 1 sklearn gender pipeline."""
    out = pd.DataFrame(index=df.index)
    clean_name = df["name"].fillna("").astype(str).str.strip()
    first = clean_name.str.split().str[0].fillna("").str.lower()

    out["first_name"] = first
    out["last_1"] = first.str[-1:]
    out["last_2"] = first.str[-2:]
    out["last_3"] = first.str[-3:]
    out["first_2"] = first.str[:2]
    out["first_name_len"] = first.str.len().astype(float)
    out["ends_with_vowel"] = first.str[-1:].isin(list("aeiouy")).astype(float)
    out["ends_with_a"] = (first.str[-1:] == "a").astype(float)
    out["ends_with_e"] = (first.str[-1:] == "e").astype(float)
    out["ends_with_y_ie"] = (first.str.endswith("y") | first.str.endswith("ie")).astype(float)
    out["company"] = df["company"].fillna("Unknown").astype(str).str.strip()
    out["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(40.0).astype(float)
    return out
