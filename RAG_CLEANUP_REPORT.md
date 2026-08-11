# PhosProcess Copilot — RAG v1 cleanup

Date: 2026-08-10

## Goal

Remove hidden/domain-authored answers and fixed answer contracts from production RAG code while preserving retrieval, explicit source locking, citations, technical terminology, and objective validation.

The governing boundary is:

- **Python may know how to retrieve and validate evidence.**
- **Python must not know what the documentary answer is supposed to say.**
- **The corpus + LLM must supply domain facts, relations, ordering, values, and completeness.**

## Removed

- Domain answer taxonomies such as `_RELATION_CONCEPTS`, `_STRICT_RELATION_CONCEPTS`, process-flow concept checklists, and conical-bottom / pump-exchanger expected-answer rules.
- Hard-coded plant values and stream identifiers used as retrieval targets (P2O5 values, line 1/5/6 rules, fixed process sequence facts).
- Deterministic answer reconstruction/pruning and the fixed five-claim answer cap.
- `max_words`, 100-word response instructions, fixed sentence/step/item counts, and fixed `[Source 1]..[Source 5]` assumptions.
- Schema-level five-source limits. Source numbering and selected-source cardinality are now dynamic; `top_k` remains a configurable retrieval budget rather than a semantic requirement.
- Automatic question-to-book mappings, source boosts, section-affinity rules, and baked-in domain source priorities. Automatic source selection is global retrieval + reranking; a hard document filter is used only for an explicit user source request.
- Legacy query-expansion tables that encoded process facts/numeric values. Legacy expansion now uses terminology equivalents only.
- Process-specific evidence-coverage recovery and process-specific reranker filters.
- Hidden conversation-state inferences such as inferring a phosphoric-acid production process merely because an evaporator was mentioned.
- Domain-word routing for deciding whether to invoke RAG. Only self-contained translation/rewrite/summarization and greetings bypass retrieval; other factual/ambiguous requests use RAG.
- Domain-specific context-window boosts for P2O5/SO4/CaSO4. Context scoring now uses generic query overlap plus generic numeric/formula/symbol signals.
- Stale `__pycache__`, `.pyc`, `.broken`, and `.before_*` artifacts.

## Preserved intentionally

- Dense retrieval, BGE sparse retrieval, BM25, RRF, reranking, hierarchical retrieval, contextual expansion and metadata.
- Global document discovery and runtime source lock.
- Explicit user-selected source modes and source-name aliases. These are document identifiers/metadata, not automatic question-to-book routing.
- Small multilingual **terminology** equivalences (for example French/Arabic technical term -> English technical term). These improve recall but do not encode relations or answers.
- Generic retrieval roles such as definition, sequence, entry, transitions, exit, conservation relation, inputs, outputs, causes/effects/actions.
- A structured LLM resolver only for ambiguous conversational references. It emits a standalone query and source intent, never an answer.
- Exact citation/source-number validation and fail-closed handling of malformed generation output.
- Technical context/output token budgets. These are runtime/model budgets, not answer-content limits.

## Isolated as evaluation-only

- Phase-10 requirement planning and semantic-audit helpers were moved to `src/phosprocess/evaluation/legacy_answer_validation_service.py`.
- Phase-12 `EvidencePlanner` was moved to `src/phosprocess/evaluation/evidence_planner.py`.
- Region-aware expansion, the stricter Phase-11 prompt experiment and their frozen outputs remain under evaluation paths for reproducibility only.
- Production imports no module from `phosprocess.evaluation`.

## Main production files changed

### RAG

- `src/phosprocess/rag/adaptive_router.py`
- `src/phosprocess/rag/citation_binding.py`
- `src/phosprocess/rag/claim_support.py`
- `src/phosprocess/rag/context_window.py`
- `src/phosprocess/rag/conversation_state.py`
- `src/phosprocess/rag/followup_resolver.py`
- `src/phosprocess/rag/generation_service.py`
- `src/phosprocess/rag/orchestrator.py`
- `src/phosprocess/rag/prompts.py`
- `src/phosprocess/rag/quality_retrieval.py`
- `src/phosprocess/rag/retrieval_service.py`
- `src/phosprocess/rag/schemas.py`
- `src/phosprocess/rag/source_policy.py`

### Retrieval

- `src/phosprocess/retrieval/context_expander.py`
- `src/phosprocess/retrieval/domain_router.py`
- `src/phosprocess/retrieval/evidence_bundle.py`
- `src/phosprocess/retrieval/evidence_coverage.py`
- `src/phosprocess/retrieval/hierarchical.py`
- `src/phosprocess/retrieval/hybrid.py`
- `src/phosprocess/retrieval/query_expansion.py`
- `src/phosprocess/retrieval/retrieval_planner.py`
- `src/phosprocess/retrieval/technical_lexicon.py`

## Target runtime architecture

```text
User question
    |
    +--> direct language task? --> LLM directly
    |
    `--> factual / documentary request
            |
            v
      standalone query / generic intent
            |
            v
      global Dense + Sparse + BM25
            |
            v
             RRF
            |
            v
          reranker
            |
            v
      document evidence ranking
            |
            v
         SOURCE LOCK
            |
            v
        deep retrieval
            |
            v
       LLM generation
            |
            v
  objective citation validation
            |
            v
        final answer
```

No Python function is allowed to inject a missing domain stage/value/relation in order to make an answer pass.

## Validation performed in the project environment

- `python -m compileall -q src` passed.
- `python -m ruff check src tests` passed.
- `python -m pytest -q` passed: **443 tests**.
- Static audit: **0** hits in active `rag/` + `retrieval/` for the audited hidden-answer facts/taxonomies (including P2O5 line/value rules, conical-bottom process requirements, pump-exchanger expected sequence, relation concept taxonomies).
- Static audit: **0** fixed answer/source-cardinality rules of the audited kinds in active `rag/` + `retrieval/` (`max_words`, fixed five claims/sources, exact-five source schema limits, 100-word prompt cap).
- Static audit: no automatic `_profile_scores` / `_DOMAIN_PRIMARY_DOCUMENTS` / baked question-to-book route table remains.
- Smoke check passed for generic process-flow retrieval (`flow path`, structural sequence roles) and legitimate terminology expansion for momentum transport.
- Stale Python/cache/backup artifacts and proven-dead temporary patch/diagnostic scripts were removed; frozen evaluation datasets and reports were preserved.
- Real Dense, BGE sparse, BM25, reranker and qwen3:8b loading passed during the final latency profile.
- The production architecture guard asserts that active generation contains no requirement planner, semantic verifier, citation repair loop, deterministic answer builder, or evaluation import.

Final benchmark and freeze details belong to `RAG_V1_REPORT.md`.
