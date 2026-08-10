"""Deterministic gates for Phase-10 post-generation evaluation."""

from phosprocess.evaluation.generation_baseline_analysis_v01 import (
    _detected_language,
    evidence_coverage,
    extract_atomic_claims,
)
from phosprocess.evaluation.generation_baseline_report_v01 import _ratio
from phosprocess.evaluation.generation_manual_review_v01 import (
    _best_evidence_windows,
)


def test_claim_extraction_only_splits_and_preserves_citations() -> None:
    claims = extract_atomic_claims(
        "Pressure rises [Source 1]; boiling temperature rises [Source 2]."
    )
    assert [item["claim_text"] for item in claims] == [
        "Pressure rises",
        "boiling temperature rises .",
    ]
    assert [item["citation_numbers"] for item in claims] == [[1], [2]]
    assert all("support" not in item for item in claims)


def test_evidence_coverage_respects_alternative_and_complementary_sets() -> None:
    sets = [
        {"type": "alternative", "chunk_ids": ["a", "b"]},
        {"type": "complementary", "groups": [["c", "c2"], ["d"]]},
    ]
    assert evidence_coverage(sets, {"b"}) == (True, 1.0)
    assert evidence_coverage(sets, {"c"}) == (False, 0.5)
    assert evidence_coverage(sets, {"c", "d"}) == (True, 1.0)


def test_language_detection_covers_french_english_and_arabic() -> None:
    assert _detected_language("La pompe est dans la boucle et le débit est stable.") == "fr"
    assert _detected_language("The pump is in the loop and the flow is stable.") == "en"
    assert _detected_language("تعمل المضخة داخل الحلقة للحفاظ على التدفق.") == "ar"


def test_review_packet_selects_wording_without_judging_support() -> None:
    windows = _best_evidence_windows(
        "Fouling increases thermal resistance.",
        "The pump circulates acid. Fouling increases the thermal resistance.",
        limit=1,
    )
    assert windows == ["Fouling increases the thermal resistance."]


def test_report_ratio_handles_empty_denominator() -> None:
    assert _ratio(0, 0) == 0.0
    assert _ratio(1, 2) == 0.5
