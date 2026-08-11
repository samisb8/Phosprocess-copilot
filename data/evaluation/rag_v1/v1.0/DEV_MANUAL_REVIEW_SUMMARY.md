# RAG V1 — DEV Manual Documentary Review

Scope reviewed:
- 52 generated DEV records
- 307 atomic claims
- 45 primary DEV questions
- Evidence basis: frozen question packet, documentary justifications, cited EvidenceBundle windows, and full generated answers.
- No semantic LLM judge was introduced into production.

## Primary DEV manual metrics

- Answer success: 32/45 = 71.11%
- Manual evidence availability: 39/45 = 86.67%
- Evidence use among manually available cases:
  - YES: 29
  - PARTIAL: 9
  - NO: 1

Completeness:
- COMPLETE: 26
- MOSTLY_COMPLETE: 6
- PARTIAL: 8
- MISSED: 5

Claim support:
- SUPPORTED: 218 (77.03%)
- PARTIALLY_SUPPORTED: 56 (19.79%)
- UNSUPPORTED: 9 (3.18%)

Citation support among cited primary claims:
- Strict precision (SUPPORTED): 71.88%
- SUPPORTED or PARTIALLY_SUPPORTED: 94.79%

## Important DEV findings

Strong:
- Citation references remain syntactically valid.
- Absent-corpus refusals are correct.
- FR/EN conversational follow-ups resolve to useful documentary answers.
- The final Qwen3:8B path is substantially faster than the earlier Phase-10 path.

Known weaknesses retained for V1 documentation:
- CE051: scaling question retrieves/uses corrosion material instead of the requested scaling effects.
- CE066: 27→54% Becker example replaces the requested plant-specific 29→54% route.
- DQ003: reaction-slurry recirculation replaces forced-evaporator recirculation.
- DQ044 as a standalone fragment remains unresolved without history and drifts to distillation/general manipulated-variable material.
- DQ048 Arabic routing uses refrigeration-evaporator material; Arabic conversation behavior is best-effort and is not part of the FR/EN V1 conversational acceptance gate.
- PROCESS_FLOW_ACCEPTANCE: individual OCP statements are largely grounded, but two operating descriptions are concatenated; the answer backtracks after product storage and mixes ~75 mmHg with 60 Torr.
- DQ021 reaches the generation limit and is reduced to the last cited boundary without retry; the substantive fouling answer is preserved but contains substantial repetition.

These labels are intended for the frozen DEV evaluator workflow, not as production logic.
