"""Prepare chunking ablation exclusively on the domain-quality DEV set."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from phosprocess.evaluation.domain_quality import (
    DEFAULT_DOMAIN_QUALITY_DIRECTORY,
    load_domain_questions,
    validate_domain_questions,
)

ABLATION_CONFIGS = {
    "A": {"target": 300, "maximum": 420, "overlap": 50},
    "B": {"target": 420, "maximum": 560, "overlap": 70},
    "C": {"target": 550, "maximum": 700, "overlap": 90},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation de chunking sur le DEV métier uniquement."
    )
    parser.add_argument(
        "--config",
        choices=["all", *ABLATION_CONFIGS],
        default="all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = validate_domain_questions(load_domain_questions())
    selected = (
        ABLATION_CONFIGS
        if arguments.config == "all"
        else {arguments.config: ABLATION_CONFIGS[arguments.config]}
    )
    review = (
        DEFAULT_DOMAIN_QUALITY_DIRECTORY / "human_review_template.csv"
    ).read_text(encoding="utf-8")
    has_validated_evidence = ",validated," in review

    print(f"Questions DEV métier validées : {summary.question_count}")

    for name, config in selected.items():
        print(
            f"{name}: target={config['target']} max={config['maximum']} "
            f"overlap={config['overlap']}"
        )

    if not has_validated_evidence:
        print(
            "Aucune variante gagnante : les preuves sont encore "
            "needs_human_review."
        )
        print("Benchmark TEST historique utilisé : non")
        return 0

    raise SystemExit(
        "L'évaluation retrieval après revue sera activée dans une étape dédiée."
    )


if __name__ == "__main__":
    raise SystemExit(main())
