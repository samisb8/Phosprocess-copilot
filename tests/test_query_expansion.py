from __future__ import annotations

import pytest

from phosprocess.retrieval.hybrid import (
    expand_lexical_query,
)


def test_phosphoric_v1_is_preserved() -> None:
    query = "Quel est le rapport de recirculation externe ?"

    expanded = expand_lexical_query(
        query,
        version="phosphoric_v1",
    )

    assert "external recirculation ratio" in expanded
    assert "sludge removal losses" not in expanded


def test_phosphoric_v2_expands_p2o5_losses() -> None:
    query = (
        "Quelles categories de pertes de P2O5 "
        "doivent etre prises en compte ?"
    )

    expanded = expand_lexical_query(
        query,
        version="phosphoric_v2",
    )

    expected = [
        "P2O5 losses",
        "co-crystallized losses",
        "lattice losses",
        "cake impregnation losses",
        "unattacked P2O5",
        "mechanical losses",
        "sludge removal losses",
    ]

    for phrase in expected:
        assert phrase in expanded


def test_phosphoric_v2_expands_intermediate_clarification() -> None:
    query = (
        "Pourquoi une clarification interm\u00e9diaire "
        "vers 40 % P2O5 est-elle preferable ?"
    )

    expanded = expand_lexical_query(
        query,
        version="phosphoric_v2",
    )

    expected = [
        "intermediate clearing",
        "intermediate clarification",
        "intermediate settling",
        "40% P2O5",
    ]

    for phrase in expected:
        assert phrase in expanded


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Version d'expansion inconnue",
    ):
        expand_lexical_query(
            "test",
            version="unknown",
        )
