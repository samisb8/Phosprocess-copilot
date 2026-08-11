# dev_best_v3 — configuration DEV figée

## Décision

- Variante retenue : `lexical_safeguard_001`.
- Décision fondée exclusivement sur les artefacts de robustesse DEV.
- Aucun TEST utilisé, lu ou exécuté.
- Une seule variante est marquée gagnante.

## Métriques DEV finales

- Candidate Recall@20 : 1.0000
- Evidence Recall@5 : 1.0000
- Hit@5 : 1.0000
- MRR@5 : 0.7885
- Hit@1 : 0.6875

## Paramètres

- candidate_k : 20
- dense_candidates : 20
- bm25_candidates : 20
- query_expansion : true
- top_k : 5
- lexical_slots : 1
- reranker_leading_slots : 4
- lexical_source : bm25
- fallback : next_reranker_result

## Comparaison des variantes

| Variante | Candidate Recall@20 | Evidence Recall@5 | Hit@5 | MRR@5 | Hit@1 | Régressions |
|---|---:|---:|---:|---:|---:|---:|
| lexical_safeguard_001 | 1.0000 | 1.0000 | 1.0000 | 0.7885 | 0.6875 | 0 |
| strict_lexical_slots_0 | 1.0000 | 0.9375 | 0.9375 | 0.7760 | 0.6875 | 0 |
| permissive_lexical_slots_2 | 1.0000 | 0.9375 | 0.9375 | 0.7760 | 0.6875 | 1 |

## Intégrité

- Identité SHA-256 du snapshot : `090C0CC191317DF0CBF068725144FEA567995EFD0E7EAE64EE1EB8DD9006A4EE`.
- Les copies ont été comparées octet par octet aux sources.
- Les composants sources v2 n'ont pas été modifiés.
