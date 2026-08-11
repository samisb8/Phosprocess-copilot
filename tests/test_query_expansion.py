"""Legacy lexical expansion must not inject documentary answer content."""

from __future__ import annotations

import pytest

from phosprocess.retrieval.hybrid import expand_lexical_query


@pytest.mark.parametrize("version", ["phosphoric_v1", "phosphoric_v2"])
def test_legacy_domain_versions_are_safe_passthroughs(version: str) -> None:
    query = "Quelles informations documentaires répondent à cette question ?"
    assert expand_lexical_query(query, version=version) == query


def test_legacy_expansion_does_not_inject_expected_answer_facts() -> None:
    expanded = expand_lexical_query(
        "Décris le trajet et les pertes documentées.",
        version="phosphoric_v2",
    ).casefold()
    for forbidden in (
        "product outlet",
        "conical bottom",
        "cake impregnation losses",
        "sludge removal losses",
        "40% p2o5",
    ):
        assert forbidden not in expanded


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="Version d'expansion inconnue"):
        expand_lexical_query("test", version="unknown")
