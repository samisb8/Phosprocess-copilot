# RAG V1 — FINAL HOLDOUT Manual Review

Reviewed scope:
- 19 FINAL HOLDOUT primary questions
- 4 FINAL HOLDOUT conversational follow-up turns
- 110 newly extracted holdout claims
- Existing 52 DEV annotations were preserved unchanged.

## Holdout primary results

- Answer success: 13/19 = 68.42%
- Manual evidence availability: 16/19 = 84.21%
- Completeness:
  - COMPLETE: 10
  - MOSTLY_COMPLETE: 3
  - PARTIAL: 3
  - MISSED: 3
- Holdout primary claim support (90 claims):
  - SUPPORTED: 75 = 83.33%
  - PARTIALLY_SUPPORTED: 13 = 14.44%
  - UNSUPPORTED: 2 = 2.22%

## Combined DEV + FINAL HOLDOUT primary results

- Answer success: 45/64 = 70.31%
- Manual evidence availability: 55/64 = 85.94%
- Completeness:
  - COMPLETE: 36
  - MOSTLY_COMPLETE: 9
  - PARTIAL: 11
  - MISSED: 8
- Important claim support (373 claims):
  - SUPPORTED: 293 = 78.55%
  - PARTIALLY_SUPPORTED: 69 = 18.50%
  - UNSUPPORTED: 11 = 2.95%
- Cited important claims: 249
- Strict manual citation precision: 73.09%
- Supported-or-partially-supported citation precision: 94.78%

## Important FINAL HOLDOUT findings

- DQ002: retrieval misses the forced-circulation pump/heating route; generation narrates upstream phosphoric-acid manufacture instead.
- DQ027: the Bird pipe momentum-balance route is absent from final context; the answer relies on unrelated OCP force/energy passages.
- DQ028: laminar/turbulent/Reynolds evidence is present, but generation expands it into a long unsupported list of non-regime attributes.
- DQ037: Becker evidence is present, but the answer incorrectly says shorter growth time can increase crystal size.
- DQ042 primary: unresolved standalone fragment drifts to OCP circulation-flow sensitivity instead of the intended control-disturbance relation.
- DQ047 primary: the correct acid-after-exchanger route is present, but the generator selects the steam-condensate path.
- CONV01_T2_DQ002: resolver succeeds, but generation ignores the forced-circulation loop evidence and answers with upstream clarification/concentration steps.
- CONV05_T2_DQ047: resolver correctly makes the acid referent explicit, yet generation still answers with the steam path.
- CONV03_T2_DQ042: relevant Seborg evidence is present, but the answer reverses the causal direction of the resolved question.

These labels are evaluator artifacts only. They must not be converted into production rules or hidden answer contracts.
