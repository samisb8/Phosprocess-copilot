"""Deterministic evidence coverage vocabulary tests."""

from __future__ import annotations

from phosprocess.retrieval.evidence_coverage import (
    coverage_keys_for_text,
    provenance_with_evidence_roles,
)


def test_process_flow_detects_weak_acid_feed_entry() -> None:
    text = (
        "Weak phosphoric acid is introduced into the circulation loop "
        "of the forced-circulation evaporator."
    )

    keys = coverage_keys_for_text(text, "process_flow")

    assert "feed_inlet" in keys


def test_process_flow_detects_feed_line_wording() -> None:
    text = (
        "The dilute acid feed line supplies the evaporator before "
        "the circulating pump."
    )

    keys = coverage_keys_for_text(text, "process_flow")

    assert "feed_inlet" in keys


def test_process_flow_detects_french_feed_entry() -> None:
    text = (
        "L'acide phosphorique dilué est introduit dans le circuit de "
        "circulation de l'évaporateur."
    )

    keys = coverage_keys_for_text(text, "process_flow")

    assert "feed_inlet" in keys


def test_weak_acid_without_entry_is_not_sufficient() -> None:
    text = "Weak phosphoric acid properties are listed in this table."

    keys = coverage_keys_for_text(text, "process_flow")

    assert "feed_inlet" not in keys


def test_process_flow_detects_concentrated_acid_withdrawal() -> None:
    text = (
        "A portion of the concentrated phosphoric acid is withdrawn "
        "from the evaporator circulation loop as product."
    )

    keys = coverage_keys_for_text(text, "process_flow")

    assert "product_outlet" in keys


def test_process_flow_detects_product_sent_to_storage() -> None:
    text = (
        "The product acid leaves the vapor body and is sent to storage."
    )

    keys = coverage_keys_for_text(text, "process_flow")

    assert "product_outlet" in keys


def test_concentrated_acid_without_exit_is_not_sufficient() -> None:
    text = "Concentrated phosphoric acid properties are listed in this table."

    keys = coverage_keys_for_text(text, "process_flow")

    assert "product_outlet" not in keys


def test_process_flow_detects_conical_bottom_as_mandatory_evidence() -> None:
    text = (
        "The cycling acid leaves the vapor body through a conical bottom."
    )

    keys = coverage_keys_for_text(text, "process_flow")

    assert "conical_bottom" in keys


def test_process_flow_provenance_keeps_source_and_all_detected_roles() -> None:
    text = (
        "The acid enters through the feed inlet. The cycling acid leaves "
        "the vapor body through a conical bottom. The circulation pump "
        "forces it through the heat exchanger back to the flash chamber. "
        "The concentrated product acid is withdrawn at the product outlet."
    )

    roles = coverage_keys_for_text(text, "process_flow")
    provenance = provenance_with_evidence_roles("coverage_guard", roles)

    assert provenance == (
        "coverage_guard;evidence_roles:feed_inlet,conical_bottom,"
        "pump_heat_exchanger,vapor_body,recirculation,product_outlet"
    )
