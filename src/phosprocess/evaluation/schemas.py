"""Schémas et configuration du benchmark de retrieval."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class QueryLanguage(StrEnum):
    """Langue principale de la question."""

    FR = "fr"
    EN = "en"
    MIXED = "mixed"


class QueryCategory(StrEnum):
    """Catégorie fonctionnelle d'une question."""

    EXACT_NUMERIC = "exact_numeric"
    CAUSAL_MECHANISM = "causal_mechanism"
    PROCESS_DESCRIPTION = "process_description"
    OPERATOR_DIAGNOSIS = "operator_diagnosis"
    PROCESS_COMPARISON = "process_comparison"
    TABLE_DATA = "table_data"
    IMPURITIES_LOSSES_CORROSION = "impurities_losses_corrosion"
    UNANSWERABLE = "unanswerable"


class QueryDifficulty(StrEnum):
    """Difficulté estimée de retrieval."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DatasetSplit(StrEnum):
    """Split du benchmark."""

    DEV = "dev"
    TEST = "test"


class JudgmentStatus(StrEnum):
    """État d'un jugement de pertinence."""

    DRAFT = "draft"
    VERIFIED = "verified"
    ADJUDICATED = "adjudicated"


class EvaluationQuery(BaseModel):
    """Question individuelle du benchmark."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query_id: str = Field(
        pattern=r"^Q\d{3}$",
    )

    question: str = Field(
        min_length=5,
    )

    language: QueryLanguage
    category: QueryCategory
    difficulty: QueryDifficulty

    answerable: bool
    split: DatasetSplit

    question_family_id: str = Field(
        min_length=3,
        pattern=r"^[a-z0-9_]+$",
    )

    expected_answer: str | None = None
    notes: str = ""

    reference_documents: list[str] = Field(
        default_factory=list,
    )

    @field_validator("reference_documents")
    @classmethod
    def validate_reference_documents(
        cls,
        values: list[str],
    ) -> list[str]:
        """Nettoyer et dédupliquer les références."""

        normalized = [value.strip() for value in values if value.strip()]

        if len(normalized) != len(set(normalized)):
            raise ValueError("reference_documents contient des doublons.")

        return normalized

    @model_validator(mode="after")
    def validate_answerability(
        self,
    ) -> EvaluationQuery:
        """Vérifier la cohérence réponse/catégorie."""

        if self.answerable:
            if self.category == QueryCategory.UNANSWERABLE:
                raise ValueError(
                    "Une question répondable ne peut pas avoir la catégorie unanswerable."
                )

            if not self.expected_answer:
                raise ValueError("Une question répondable doit contenir expected_answer.")

            if not self.reference_documents:
                raise ValueError(
                    "Une question répondable doit contenir au moins un document de référence."
                )

        else:
            if self.category != QueryCategory.UNANSWERABLE:
                raise ValueError(
                    "Une question non répondable doit utiliser la catégorie unanswerable."
                )

            if self.expected_answer is not None:
                raise ValueError(
                    "Une question non répondable ne doit pas contenir expected_answer."
                )

            if self.reference_documents:
                raise ValueError(
                    "Une question non répondable ne doit pas déclarer de document de référence."
                )

        return self


class RelevanceJudgment(BaseModel):
    """Jugement query-chunk gradué de 0 à 3."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query_id: str = Field(
        pattern=r"^Q\d{3}$",
    )

    chunk_id: str = Field(
        min_length=5,
    )

    relevance: int = Field(
        ge=0,
        le=3,
    )

    rationale: str = Field(
        min_length=3,
    )

    assessor_id: str = Field(
        min_length=2,
    )

    status: JudgmentStatus = JudgmentStatus.DRAFT

    judged_at_utc: datetime


class DatasetConfig(BaseModel):
    """Chemins et identité du dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    schema_version: str

    output_directory: str

    queries_filename: str
    judgments_filename: str
    manifest_filename: str
    validation_report_filename: str


class ExpectedConfig(BaseModel):
    """Comptages attendus."""

    model_config = ConfigDict(extra="forbid")

    total_queries: int = Field(gt=0)
    answerable_queries: int = Field(ge=0)
    unanswerable_queries: int = Field(ge=0)

    splits: dict[str, int]
    categories: dict[str, int]

    answerable_languages: dict[str, int]
    unanswerable_languages: dict[str, int]
    all_languages: dict[str, int]

    @model_validator(mode="after")
    def validate_language_counts(
        self,
    ) -> ExpectedConfig:
        """Contrôler les répartitions linguistiques."""

        expected_keys = {
            QueryLanguage.FR.value,
            QueryLanguage.EN.value,
            QueryLanguage.MIXED.value,
        }

        language_groups = {
            "answerable_languages": (self.answerable_languages),
            "unanswerable_languages": (self.unanswerable_languages),
            "all_languages": self.all_languages,
        }

        for group_name, counts in language_groups.items():
            if set(counts) != expected_keys:
                raise ValueError(f"{group_name} doit contenir exactement 'fr', 'en' et 'mixed'.")

            if any(count < 0 for count in counts.values()):
                raise ValueError(f"{group_name} contient un comptage négatif.")

        if sum(self.answerable_languages.values()) != self.answerable_queries:
            raise ValueError(
                "La somme de answerable_languages ne correspond pas à answerable_queries."
            )

        if sum(self.unanswerable_languages.values()) != self.unanswerable_queries:
            raise ValueError(
                "La somme de unanswerable_languages ne correspond pas à unanswerable_queries."
            )

        if sum(self.all_languages.values()) != self.total_queries:
            raise ValueError(
                "La somme de all_languages ne correspond pas au nombre total de questions."
            )

        for language in expected_keys:
            expected_total = (
                self.answerable_languages[language] + self.unanswerable_languages[language]
            )

            if self.all_languages[language] != expected_total:
                raise ValueError(f"Répartition linguistique incohérente pour {language}.")

        return self


class AnnotationConfig(BaseModel):
    """Règles d'annotation."""

    model_config = ConfigDict(extra="forbid")

    relevance_minimum: int
    relevance_maximum: int
    binary_relevance_threshold: int
    require_rationale: bool

    labels: dict[int, str]

    @model_validator(mode="after")
    def validate_relevance_scale(
        self,
    ) -> AnnotationConfig:
        """Contrôler l'échelle de pertinence."""

        if self.relevance_minimum != 0:
            raise ValueError("La pertinence minimale doit être 0.")

        if self.relevance_maximum != 3:
            raise ValueError("La pertinence maximale doit être 3.")

        if not (
            self.relevance_minimum <= self.binary_relevance_threshold <= self.relevance_maximum
        ):
            raise ValueError("binary_relevance_threshold est hors limites.")

        expected_labels = set(
            range(
                self.relevance_minimum,
                self.relevance_maximum + 1,
            )
        )

        if set(self.labels) != expected_labels:
            raise ValueError("Les labels doivent couvrir exactement 0, 1, 2, 3.")

        return self


class PoolingConfig(BaseModel):
    """Profondeur de pooling par système."""

    model_config = ConfigDict(extra="forbid")

    dense_depth: int = Field(gt=0)
    bm25_depth: int = Field(gt=0)
    hybrid_depth: int = Field(gt=0)
    reranker_depth: int = Field(gt=0)


class MetricsConfig(BaseModel):
    """Métriques qui seront calculées plus tard."""

    model_config = ConfigDict(extra="forbid")

    candidate_retrieval: list[str]
    final_ranking: list[str]


class EvaluationConfig(BaseModel):
    """Configuration racine."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    expected: ExpectedConfig
    annotation: AnnotationConfig
    pooling: PoolingConfig
    metrics: MetricsConfig

    pipeline_version: str


def load_evaluation_config(
    config_path: Path,
) -> EvaluationConfig:
    """Charger et valider evaluation.yaml."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration introuvable : {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(raw_config, dict):
        raise ValueError("Le fichier evaluation.yaml doit contenir un objet YAML.")

    return EvaluationConfig.model_validate(raw_config)
