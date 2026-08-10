# PhosProcess Copilot — RAG V1 Final Report

Freeze date: 2026-08-10  
RAG version: 1.0.0  
Status: READY / FROZEN

## 1. Final architecture

```text
User query
  -> conversation/query resolution
  -> standalone question
  -> explicit-source resolution OR global document discovery
  -> source lock
  -> hybrid retrieval:
       BGE-M3 dense
       + BGE-M3 sparse
       + BM25
  -> RRF / dynamic ColBERT fusion
  -> BGE-Reranker-v2-M3
  -> Context Engine
  -> EvidenceBundles
  -> one Qwen3:8B grounded generation call
  -> objective citation / format validation
  -> final answer
```

Python owns infrastructure, orchestration, source binding, deterministic metadata checks and objective validation. The LLM owns language understanding and answer generation. Documentary content is the only source of domain facts.

The active production path contains no deterministic expected answer, no hidden business sequence, no semantic judge, no EvidencePlanner, no verifier and no citation-repair LLM loop.

## 2. Knowledge base

Active knowledge base:

- Version: `kb_quality_20260809_015038`
- Documents: 8 audited PDFs
- Pages: 6,615
- Child chunks: 24,085
- Parser: Docling
- Representation: hierarchical document/section/chunk metadata with parent/neighbor relationships

RAG V1 freezes the existing corpus and indexes. No re-ingestion, rechunking, embedding replacement or index redesign is part of the release.

## 3. Retrieval stack

Frozen retrieval configuration:

- Variant: `lexical_safeguard_001`
- Dense model: `BAAI/bge-m3`
- Sparse model: `BAAI/bge-m3`
- Dense dimension: 1,024
- BM25 tokenizer: `technical_v1`
- Candidate target: 20
- Dense candidates: 20
- Sparse candidates: 50
- BM25 candidates: 20
- Query expansion: enabled
- Final anchors: 5
- Lexical safeguard slots: 1
- Leading reranker slots: 4
- Fusion: RRF, `k=60`, with dynamic ColBERT contribution
- Reranker: `BAAI/bge-reranker-v2-m3`
- Reranker candidate depth: 30

Automatic queries perform global documentary discovery before source lock. Explicit user source constraints always win. Once a document is locked, cross-document evidence leakage is forbidden.

Frozen retrieval research was completed before the V1 generation freeze. No additional retrieval research was performed after the holdout was opened.

## 4. Context Engine

The Context Engine packages retrieved evidence into `EvidenceBundle` objects while preserving:

- document identity,
- page,
- section,
- anchor chunk,
- supporting chunks,
- selection provenance.

Frozen packing limits:

- maximum 650 documentary tokens per source,
- maximum 2,600 documentary tokens overall,
- neighbor window: 1,
- parent inclusion: conditional.

Conversation history is used for query understanding only and is never treated as documentary evidence.

## 5. Generator

Final generator: `qwen3:8b`

Frozen parameters:

- prompt family: Phase-10 baseline
- temperature: 0.1
- seed: 0
- context size: 8,192
- maximum output: 1,024 tokens
- thinking: disabled
- keep-alive: 30 minutes
- one answer-generation call per documentary answer

A controlled insufficiency answer is used when the supplied evidence cannot support an affirmative answer.

The `qwen3:14b` capability experiment was rejected for RAG V1 because it was materially slower and degraded citation behavior on the frozen DEV replay. The final generator therefore remains `qwen3:8b`.

## 6. Objective validation

Production validation checks only objective properties:

- citation syntax,
- cited source-number membership,
- source-lock consistency,
- controlled insufficiency format.

Production does not use:

- semantic correctness judge,
- evidence judge,
- LLM verifier,
- repair loop,
- deterministic expected answers.

Malformed affirmative output fails closed rather than being repaired by another generation call.

## 7. Conversation behavior

The conversation resolver emits a strict structured query representation and is used only when a follow-up requires contextual resolution.

History is used to understand references, not as factual evidence.

French and English conversational behavior form the supported V1 conversational acceptance scope. Arabic remains best-effort and is not a conversational release gate for V1.

The final holdout shows that query resolution can succeed while answer generation can still select the wrong documentary branch. This is retained as a measured V1 limitation rather than converted into hidden deterministic rules.

## 8. Evaluation protocol

The final evaluation contains:

- 75 generated records total,
- 64 primary benchmark questions,
- 52 DEV records,
- 23 FINAL HOLDOUT records,
- 19 FINAL HOLDOUT primary questions,
- 4 FINAL HOLDOUT follow-up turns.

The FINAL HOLDOUT was opened exactly once after:

- production baseline freeze,
- DEV completion,
- manual DEV annotation,
- evaluator freeze.

No production code, prompt, retrieval configuration or evaluator policy was changed after holdout opening.

No semantic LLM judge was used.

## 9. Freeze integrity

Production baseline SHA-256:

`5c7bfba970cdc1a14e928fffe823c8bceb87701eb8eefe4f4e3f4f46cfcbf16e`

Evaluator SHA-256:

`648e99fc7df3f6f8cbadd516506e8f5f895a5240ace5e19c1b3ebfe89c1f850f`

Evaluation protocol SHA-256:

`1a5a3119360766cb4767579e1cb7d368e8b13232db0023422ed10829b6d91cfa`

Evaluator-freeze SHA-256:

`5d7b5b17cab126a55b6670995b5be6ee75fe8e109f3358f0594be309bb24dcaa`

FINAL HOLDOUT opening count: 1.

All 19 primary holdout records completed successfully at execution level with zero runner errors.

## 10. Final objective metrics

Across all 75 generated records:

| Metric | Result |
|---|---:|
| Citation validity | 100.00% |
| Citation coverage | 67.87% |
| Numeric grounding | 72.83% |
| Unit grounding | 78.72% |
| Correct-language rate | 98.67% |
| Exact gold evidence available in final context, primary set | 57.81% |
| Context-packing misses | 6 |
| Citation repair calls | 0 |

`citation_validity` measures citation syntax/membership validity. It does not mean every cited claim is semantically entailed by the cited evidence.

## 11. Final manual primary metrics

Manual documentary review covers all final records and 417 extracted claims.

Primary benchmark: 64 questions.

### Answer success

- DEV: 32/45 = 71.11%
- FINAL HOLDOUT: 13/19 = 68.42%
- Combined: 45/64 = 70.31%

The holdout therefore shows no major DEV-to-holdout collapse.

### Manual evidence availability

- FINAL combined primary: 55/64 = 85.94%

This manual measure is higher than exact-gold context availability because equivalent documentary evidence can support an answer even when the exact frozen gold evidence set is not present.

### Completeness

| Class | Count |
|---|---:|
| COMPLETE | 36 |
| MOSTLY_COMPLETE | 9 |
| PARTIAL | 11 |
| MISSED | 8 |

### Primary claim support

373 primary claims were manually reviewed:

| Support label | Count | Rate |
|---|---:|---:|
| SUPPORTED | 293 | 78.55% |
| PARTIALLY_SUPPORTED | 69 | 18.50% |
| UNSUPPORTED | 11 | 2.95% |

For cited primary claims, manual citation precision is approximately:

- strict `SUPPORTED`: 73.09%
- `SUPPORTED` or `PARTIALLY_SUPPORTED`: 94.78%

## 12. Absent-corpus behavior

The three dedicated absent-corpus DEV tests fail closed correctly.

No fallback business answer is injected when the corpus does not support the requested information.

Absent-corpus acceptance: 3/3.

## 13. Process-flow behavior

The dedicated forced-circulation phosphoric-acid process-flow case executes successfully at system level and retrieves relevant OCP material, but strict semantic process-flow quality is not fully achieved.

Observed limitation:

- the generated answer concatenates distinct operating descriptions,
- it backtracks after the product-storage portion,
- it mixes approximately 75 mmHg and 60 Torr contexts,
- it does not express the forced-circulation loop as cleanly as required.

This limitation is documented rather than repaired with a deterministic process template. Hard-coded industrial sequences are forbidden in production.

## 14. Final holdout findings

Important retained V1 limitations include:

- `DQ002`: the final context misses the required forced-circulation pump/heating route; the answer narrates upstream phosphoric-acid processing instead.
- `DQ027`: the intended Bird pipe momentum-balance evidence is absent from final context.
- `DQ028`: relevant flow-regime evidence is present but generation expands it into unsupported non-regime attributes.
- `DQ037`: relevant Becker evidence is present, but generation makes an incorrect statement about shorter growth time increasing crystal size.
- `DQ042`: the standalone fragment is unresolved in the primary benchmark and retrieval drifts away from the intended control-disturbance relation.
- `DQ047`: the correct acid-after-exchanger evidence is present, but generation selects the steam/condensate path.
- `CONV01_T2_DQ002`: resolver identifies the forced-circulation evaporator, but generation answers with upstream clarification/concentration steps.
- `CONV03_T2_DQ042`: relevant control evidence is present, but the answer reverses the causal direction of the resolved question.
- `CONV05_T2_DQ047`: resolver correctly makes the acid referent explicit, yet generation answers with the steam path.
- Arabic conversational resolution is best-effort and is outside the V1 FR/EN conversational acceptance gate.

These findings are evaluation results. They must not be translated into question-specific production rules.

## 15. Latency

Final 75-record system profile:

| Metric | Result |
|---|---:|
| Median end-to-end latency | 29.34 s |
| Mean end-to-end latency | 35.59 s |
| Median first-token latency | 12.09 s |
| Median input size | 2,674 tokens |
| Median documentary context | 2,558 tokens |
| Median generated output | 182 tokens |

The earlier Phase-10 baseline was approximately:

- 93.63 s median end-to-end,
- 54.45 s median TTFT.

The final production path therefore materially reduces latency after removal of inactive planner/audit/repair work from the request path.

Local inference remains hardware-limited by the 8 GB GPU configuration.

## 16. Rejected or inactive experiments

The following components are not part of RAG V1 production:

- Phase-11 stricter generation prompt,
- Phase-12 EvidencePlanner,
- Evidence Judge,
- semantic verifier,
- citation-repair loop,
- region-aware candidate expansion,
- deterministic answer contracts,
- hidden expected sequences,
- GraphRAG,
- PageIndex,
- agentic recursive retrieval,
- new embeddings,
- new reranker,
- rechunking,
- fine-tuning.

Legacy experiment code may remain under evaluation-only namespaces for reproducibility, but these components are not imported into the active production answer path.

## 17. Final software gates

Final release checks on the frozen repository state:

- `python -m compileall -q src tests`: PASS
- `ruff check src tests`: PASS
- full `pytest -q`: PASS
- production baseline verification: PASS

The only reported test warning is a Starlette/httpx deprecation warning from the testing dependency stack. It is not a RAG correctness blocker.

## 18. Release decision

**RAG V1 READY**

The release is not defined as perfect answer quality. It is defined as a frozen, reproducible and measured RAG system whose remaining weaknesses are known, visible and not hidden behind hard-coded answers, semantic repair loops or benchmark-specific production logic.

The final system satisfies the architectural boundary:

**Python = infrastructure/orchestration/objective validation.  
LLM = semantic understanding/generation.  
Documents = business knowledge.**

No further RAG research is authorized for V1.

Future improvements belong to a separate V2 research backlog and must start from a newly defined evaluation protocol rather than modifying this frozen release after seeing the FINAL HOLDOUT.
