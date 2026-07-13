"""
Fraud-decision engine for facial de-duplication.

Pure, dependency-free logic (only the standard library) so it is trivially
unit-testable and has no I/O. Given a face-similarity score and the identity
metadata of the best gallery match, it decides *whether a duplicate face is
being enrolled under a different identity* — the signature of duplicate-account,
mule-account and synthetic-identity fraud at KYC onboarding.

All similarity scores are in the app's rescaled [0, 1] space
(``similarity = (2 - cosine_distance) / 2``; see redis_service.py) where 1.0 is
an identical face and ~0.5 is an unrelated face.

Verdicts
--------
CLEAR                           No sufficiently similar face in the gallery -> safe to enroll.
REVIEW                          Borderline face match, or a strong match we cannot attribute to
                                an identity -> route to a human reviewer.
DUPLICATE_SAME_IDENTITY         Same face AND same government identity -> legitimate re-KYC / dedupe.
FRAUD_ALERT_DIFFERENT_IDENTITY  Same face but a DIFFERENT government identity -> the fraud case.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---- Verdicts -------------------------------------------------------------
CLEAR = "CLEAR"
REVIEW = "REVIEW"
DUPLICATE_SAME_IDENTITY = "DUPLICATE_SAME_IDENTITY"
FRAUD_ALERT_DIFFERENT_IDENTITY = "FRAUD_ALERT_DIFFERENT_IDENTITY"

# Identity fields stored alongside every embedding and compared across records.
# ``id_number`` is the authoritative government identifier (PAN / Aadhaar-ref);
# ``customer_id`` is a per-application id and is expected to differ between
# legitimate applications, so it is only ever used as a "same" signal.
IDENTITY_FIELDS: List[str] = [
    "customer_id", "full_name", "id_type", "id_number", "phone", "dob",
]

# Fields whose *breadth of mismatch* feeds the risk score.
_RISK_FIELDS = ("id_number", "full_name", "phone", "id_type")


def normalize_value(field: str, value: Optional[str]) -> str:
    """Normalise a single identity value for robust equality comparison."""
    if value is None:
        return ""
    v = str(value).strip().lower()
    if field in ("id_number", "phone", "customer_id"):
        v = re.sub(r"[\s\-]", "", v)          # ignore spaces / dashes in IDs
    if field in ("full_name",):
        v = re.sub(r"\s+", " ", v)            # collapse internal whitespace
    return v


def normalize_identity(identity: Optional[Dict]) -> Dict[str, str]:
    identity = identity or {}
    return {f: normalize_value(f, identity.get(f)) for f in IDENTITY_FIELDS}


def compare_identity(query: Optional[Dict], match: Optional[Dict]) -> Dict[str, List[str]]:
    """Compare two identities field-by-field (only where both sides are present).

    Returns ``{"diffs": [...], "matches": [...]}``.
    """
    q = normalize_identity(query)
    m = normalize_identity(match)
    diffs, matches = [], []
    for f in IDENTITY_FIELDS:
        if q[f] and m[f]:                      # both present -> comparable
            (matches if q[f] == m[f] else diffs).append(f)
    return {"diffs": diffs, "matches": matches}


def identity_relation(compare: Dict[str, List[str]]) -> str:
    """Resolve the identity relationship to 'same' | 'different' | 'unknown'.

    Government id_number is authoritative; full_name is the fallback discriminator.
    A differing customer_id alone never implies fraud (new application = new id).
    """
    diffs, matches = compare["diffs"], compare["matches"]
    if "id_number" in diffs:
        return "different"
    if "id_number" in matches:
        return "same"
    if "full_name" in diffs:
        return "different"
    if "full_name" in matches:
        return "same"
    if "customer_id" in matches:
        return "same"
    return "unknown"


def is_same_identity(compare: Dict[str, List[str]]) -> bool:
    return identity_relation(compare) == "same"


def _diff_reason_codes(compare: Dict[str, List[str]]) -> List[str]:
    codes = []
    mapping = {
        "id_number": "ID_NUMBER_MISMATCH",
        "full_name": "NAME_MISMATCH",
        "phone": "PHONE_MISMATCH",
        "id_type": "ID_TYPE_MISMATCH",
        "dob": "DOB_MISMATCH",
    }
    for f, code in mapping.items():
        if f in compare["diffs"]:
            codes.append(code)
    if "id_number" in compare["matches"]:
        codes.append("SAME_ID_NUMBER")
    if "customer_id" in compare["matches"]:
        codes.append("SAME_CUSTOMER_ID")
    return codes


def classify(
    best_similarity: float,
    best_compare: Optional[Dict[str, List[str]]],
    has_match: bool,
    t_match: float,
    t_review: float,
) -> Tuple[str, List[str]]:
    """Return ``(verdict, reason_codes)`` for the best gallery match."""
    if not has_match or best_similarity < t_review:
        return CLEAR, ["NEW_FACE_NO_MATCH"]

    best_compare = best_compare or {"diffs": [], "matches": []}

    if best_similarity < t_match:
        return REVIEW, ["FACE_MATCH_BORDERLINE"] + _diff_reason_codes(best_compare)

    # Strong face match — decide same vs different identity.
    reason_codes = ["FACE_MATCH_HIGH"] + _diff_reason_codes(best_compare)
    relation = identity_relation(best_compare)
    if relation == "different":
        return FRAUD_ALERT_DIFFERENT_IDENTITY, reason_codes
    if relation == "same":
        return DUPLICATE_SAME_IDENTITY, reason_codes
    # Strong face match but no identity fields to attribute it to.
    return REVIEW, reason_codes + ["IDENTITY_UNVERIFIED"]


def risk_score(
    verdict: str,
    best_similarity: float,
    best_compare: Optional[Dict[str, List[str]]],
    t_review: float,
) -> int:
    """Map a verdict + evidence to a 0..100 risk score for the review queue."""
    best_compare = best_compare or {"diffs": [], "matches": []}
    span = max(1e-6, 1.0 - t_review)
    face_strength = min(1.0, max(0.0, (best_similarity - t_review) / span))
    n_diff = sum(1 for f in _RISK_FIELDS if f in best_compare["diffs"])
    mismatch_ratio = n_diff / len(_RISK_FIELDS)

    if verdict == FRAUD_ALERT_DIFFERENT_IDENTITY:
        # Floor of 60 (it is already a confirmed same-face/different-id hit),
        # scaled up by face strength and how many identity fields disagree.
        return min(100, round(60 + 40 * (0.7 * face_strength + 0.3 * mismatch_ratio)))
    if verdict == REVIEW:
        return min(100, round(40 + 25 * face_strength))
    if verdict == DUPLICATE_SAME_IDENTITY:
        return round(45 * face_strength)
    return 0  # CLEAR
