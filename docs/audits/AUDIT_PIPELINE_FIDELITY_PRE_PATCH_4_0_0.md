# Audit de refactoring — PhosProcess Copilot RAG v1

**Baseline auditée :** branche `fix-rag-grounding`, état fonctionnel post-Patch 3.9.2.  
**Objectif :** découper `pipeline.py` et `fidelity.py` sans modifier le comportement du RAG.  
**Patch préparé :** `4.0.0 — Behavior-Preserving RAG Refactor`.

## 1. Règle de sécurité

Le refactoring ne modifie pas :

- le retriever dense BGE-M3 ;
- le retriever sparse BGE-M3 ;
- BM25 ;
- le scoring ColBERT ;
- le reranker BGE ;
- les fichiers YAML de retrieval ;
- les index ;
- les embeddings ;
- les PDF ;
- les chunks ;
- les snapshots de la knowledge base.

Le patch touche uniquement les modules d’orchestration, de génération, de validation et de fidélité, plus un test de frontières architecturales.

## 2. État avant refactoring

### `src/phosprocess/rag/pipeline.py`

- **3 505 lignes** ;
- **33 blocs d’imports** ;
- **13 classes** ;
- **8 fonctions de module** ;
- **39 méthodes** dans `PhosProcessRAG`.

Le fichier concentre quatre responsabilités différentes :

1. configuration et cycle de vie du runtime ;
2. retrieval et politique documentaire ;
3. génération Qwen/Ollama ;
4. validation, citations et construction de la réponse publique.

### Classes présentes avant le patch

```text
RAGError
RAGConfigurationError
RAGRetrievalError
RAGGenerationError
RAGResponseValidationError
FrozenV3Config
ConversationRuntimeConfig
GenerationRuntimeConfig
WarmupRuntimeConfig
RAGRuntimeConfig
_RetrievedContext
_TimedTokenizerProxy
PhosProcessRAG
```

### Fonctions de module présentes avant le patch

```text
_controlled_fallback_for_language
_default_source_policy_config
sha256_file
_required_mapping
load_runtime_config
load_frozen_v3_config
_verify_snapshot_components
validate_question
```

### Méthodes de `PhosProcessRAG` classées par responsabilité

#### Orchestration et cycle de vie

```text
__init__
_build_retriever
knowledge_base_status
_install_embedding_timer
lifecycle_debug
_install_reranker_tokenizer_timer
warmup
answer
stream_answer
_coerce_memory_context
create_conversation_memory
_resolve_turn_source_mode
_quality_source_mode
_record_prompt_metrics
close
```

#### Retrieval et préparation du contexte

```text
_retrieve_quality
_retrieve_with_source_policy
_decide_source_policy
_active_source_filenames
_document_ids
_available_fallback_sources
_count_sufficient_preferred_chunks
_attach_source_policy
_retrieve
_prepare_context
```

#### Génération

```text
_direct_language_code
_build_direct_response
_answer_direct
_stream_direct_request
_reject_likely_truncation
_generate_json_answer
```

#### Validation et réponse publique

```text
_validate_answer
_validate_answer_with_metrics
_log_validation_rejection
_comparison_subjects
_cited_sources
_build_response
_build_sources
_excerpt
```

## 3. Dépendances actuelles de `pipeline.py`

### Conversation et routage

```text
adaptive_router.py
conversation_memory.py
conversation_state.py
followup_resolver.py
question_classifier.py
source_policy.py
domain_router.py
```

### Retrieval

```text
quality_retrieval.py
context_window.py
retrieval/hybrid.py
retrieval/v3_selection.py
reranking/reranker.py
knowledge_base/runtime.py
```

### Génération

```text
prompts.py
language.py
llm/ollama_client.py
```

### Fidélité et réponse

```text
citations.py
fidelity.py
schemas.py
observability/latency.py
```

### Problème principal

Une modification dans le retrieval, la génération ou les citations oblige à travailler dans la même classe de plus de 2 700 lignes. Cela augmente le risque de régression et rend difficile l’exposition future par FastAPI.

## 4. État avant refactoring de `fidelity.py`

### Mesures

- **3 334 lignes** ;
- **4 classes** ;
- **61 fonctions** ;
- **32 constantes ou dictionnaires de règles**.

### Classes

```text
ClaimSupportStatus
ClaimSupport
PrunedAnswer
AnswerContractResult
```

### Responsabilités mélangées

1. normalisation sémantique, nombres et unités ;
2. validation claim → chunk ;
3. découpage des phrases et liaison des citations ;
4. pruning des affirmations non soutenues ;
5. builders déterministes Becker, Bird, P₂O₅, fouling et process flow ;
6. contrats de réponse par type de question.

### Fonctions publiques importantes

```text
build_atomic_process_flow_answer
prune_unsupported_claims
build_deterministic_definition_answer
build_deterministic_balance_answer
build_deterministic_momentum_diffusion_answer
build_deterministic_fouling_answer
build_deterministic_scoped_explanation
enforce_answer_contract
evaluate_claim_support
validate_claim_support
```

## 5. Architecture cible du Patch 4.0.0

```text
src/phosprocess/rag/
├── orchestrator.py
├── retrieval_service.py
├── generation_service.py
├── answer_validation_service.py
├── pipeline.py                    # façade compatible
├── claim_support.py
├── citation_binding.py
├── deterministic_builders.py
├── answer_contracts.py
└── fidelity.py                    # façade compatible
```

## 6. Découpage exact de `pipeline.py`

### `orchestrator.py`

Responsabilités :

- configuration runtime ;
- chargement et cycle de vie ;
- warm-up ;
- conversation ;
- méthodes publiques `answer()` et `stream_answer()` ;
- orchestration des trois services.

`PhosProcessRAG` devient une composition par mixins :

```python
class PhosProcessRAG(
    RetrievalService,
    GenerationService,
    AnswerValidationService,
):
    ...
```

### `retrieval_service.py`

Contient :

```text
_RetrievedContext
RAGError et ses sous-classes
_retrieve_quality
_retrieve_with_source_policy
_decide_source_policy
_active_source_filenames
_document_ids
_available_fallback_sources
_count_sufficient_preferred_chunks
_attach_source_policy
_retrieve
_prepare_context
```

Aucun algorithme interne du retriever n’est déplacé ou modifié. Ce service appelle toujours les mêmes composants existants.

### `generation_service.py`

Contient :

```text
_controlled_fallback_for_language
_direct_language_code
_build_direct_response
_answer_direct
_stream_direct_request
_reject_likely_truncation
_generate_json_answer
```

Les prompts, Qwen3:8B, Ollama, le streaming et les règles de réparation restent identiques.

### `answer_validation_service.py`

Contient :

```text
_validate_answer
_validate_answer_with_metrics
_log_validation_rejection
_comparison_subjects
_cited_sources
_build_response
_build_sources
_excerpt
```

La liaison entre citations et sources reste inchangée.

### `pipeline.py`

Devient une façade de **47 lignes** qui réexporte les mêmes symboles publics :

```text
PhosProcessRAG
FrozenV3Config
RAGRuntimeConfig
ConversationRuntimeConfig
GenerationRuntimeConfig
WarmupRuntimeConfig
RAGError et sous-classes
load_runtime_config
load_frozen_v3_config
validate_question
```

Les imports existants comme :

```python
from phosprocess.rag.pipeline import PhosProcessRAG
```

restent donc valides.

## 7. Découpage exact de `fidelity.py`

### `claim_support.py`

Responsabilités :

- normalisation du texte ;
- concepts techniques ;
- nombres et unités ;
- couverture lexicale ;
- vérification qu’un chunk soutient une affirmation ;
- `evaluate_claim_support()` ;
- `validate_claim_support()`.

### `citation_binding.py`

Responsabilités :

- découpage des phrases ;
- lecture des `[Source N]` ;
- preuve atomique process flow ;
- héritage contrôlé des citations ;
- pruning déterministe ;
- `build_atomic_process_flow_answer()` ;
- `prune_unsupported_claims()`.

### `deterministic_builders.py`

Responsabilités :

- templates de réponse ;
- validation sémantique des stages ;
- builders Becker ;
- bilan P₂O₅ composite ;
- Bird momentum diffusion ;
- fouling ;
- explications ciblées.

### `answer_contracts.py`

Responsabilités :

- contrats definition/comparison/troubleshooting/balance ;
- contrôle des rôles obligatoires ;
- choix du builder déterministe ;
- `enforce_answer_contract()`.

### `fidelity.py`

Devient une façade de **48 lignes** et conserve les anciens imports publics.

## 8. Taille après refactoring

```text
pipeline.py                      47 lignes
orchestrator.py               1 800 lignes environ
retrieval_service.py            840 lignes environ
generation_service.py           580 lignes environ
answer_validation_service.py    340 lignes environ

fidelity.py                      48 lignes
claim_support.py                770 lignes environ
citation_binding.py             700 lignes environ
deterministic_builders.py     1 530 lignes environ
answer_contracts.py             355 lignes environ
```

Le but n’est pas de réduire le nombre total de lignes, mais de créer des frontières testables et explicites.

## 9. Compatibilité garantie par conception

Le patch conserve :

- les signatures publiques ;
- les mêmes noms de classes d’erreur ;
- les mêmes dataclasses ;
- le même ordre dans `stream_answer()` ;
- le même logger logique `phosprocess.rag.pipeline` ;
- les mêmes appels retrieval/generation/validation ;
- les mêmes builders déterministes ;
- les mêmes citations et sources ;
- les mêmes tests fonctionnels.

## 10. Test architectural ajouté

Nouveau fichier :

```text
tests/test_rag_refactor_boundaries.py
```

Il vérifie :

- la compatibilité de la façade `pipeline.py` ;
- la composition de `PhosProcessRAG` ;
- l’emplacement réel des méthodes par service ;
- la compatibilité de la façade `fidelity.py`.

## 11. Validation déjà exécutée sur la copie de travail

```text
python -m compileall -q src tests                      ✅
git apply --check sur une baseline propre              ✅
git diff --check                                       ✅
81 tests ciblés avec dépendances lourdes simulées       ✅
```

Les deux tests du snapshot gelé n’ont pas été exécutés dans l’environnement d’audit local, car les fichiers `data/evaluation/.../frozen/dev_best_v3` n’y sont pas présents. L’installateur exécutera le Pytest complet dans le projet utilisateur, où ces fichiers existent.

## 12. Validation imposée par l’installateur

Après application, l’installateur lance automatiquement :

```text
compileall
Ruff sur src et tests
tests ciblés du refactoring
Pytest complet
git diff --check
contrôle des imports de compatibilité
contrôle SHA des fichiers protégés du retriever
```

En cas d’échec, les deux fichiers originaux sont restaurés et les neuf nouveaux fichiers sont supprimés.

## 13. Risques et protections

| Risque | Protection |
|---|---|
| import externe cassé | façades `pipeline.py` et `fidelity.py` |
| ordre du streaming modifié | méthode `stream_answer()` déplacée sans réécriture |
| circular import | dépendances orientées services → modules bas niveau |
| modification du retriever | allowlist stricte + SHA avant/après |
| patch appliqué sur mauvaise version | signatures 3.9.2 + `git apply --check` |
| régression fonctionnelle | tests ciblés + suite complète + rollback |

## 14. Verdict

Le Patch 4.0.0 est un **refactoring structurel**, pas une nouvelle version fonctionnelle du RAG.

```text
Réponses attendues       identiques
Sources attendues        identiques
Citations attendues      identiques
Retriever                inchangé
Index                     inchangé
Configuration retrieval  inchangée
API Python publique      compatible
Organisation du code     améliorée
```
