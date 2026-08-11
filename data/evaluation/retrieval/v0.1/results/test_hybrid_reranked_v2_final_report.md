# Rapport final — Hybrid + BGE Reranker TEST v2

## Statut de l'expérience

- Exécution TEST : unique et terminée
- Retuning après TEST : interdit et non effectué
- Début UTC : 2026-07-26T13:56:47.950895+00:00
- Fin UTC : 2026-07-26T13:57:13.374875+00:00
- Questions TEST évaluées : 24 answerable
- Questions unanswerable exclues des métriques retrieval : 4

## Artefacts figés

- Configuration DEV : `data/evaluation/retrieval/v0.1/frozen/dev_best_v2`
- Gold TEST : `data/evaluation/retrieval/v0.1/frozen/test_gold_final_v2/gold_evidence_test_verified_auto.jsonl`
- SHA-256 du gold : `93C417B7AA6ECE47172FCC9197BCFB0AB3C84055AB782D073A739B1A64E95F99`
- Entrées gold : 28
- Answerable : 24
- Unanswerable : 4

Paramètres hérités du DEV figé :

- `candidate_k = 20`
- `dense_candidates = 20`
- `bm25_candidates = 20`
- fusion RRF avec expansion `phosphoric_v2`
- reranking BGE sur 20 candidats
- `top_k = 5`

## Résultats TEST

| Métrique | TEST | DEV figé | Écart TEST - DEV |
|---|---:|---:|---:|
| Candidate Recall@20 | 0.833 | 1.000 | -0.167 |
| Hit@1 | 0.500 | 0.688 | -0.188 |
| Hit@5 | 0.792 | 0.938 | -0.146 |
| MRR@5 | 0.628 | 0.776 | -0.148 |
| Evidence Recall@5 | 0.792 | 0.938 | -0.146 |
| Médiane reranking | 318.3 ms | 315.8 ms | +2.5 ms |
| Latence totale moyenne | 430.4 ms | 483.6 ms | -53.1 ms |

Comptages :

- Gold présent dans les 20 candidats : 20/24
- Gold au rang 1 final : 12/24
- Gold dans le top-5 final : 19/24

## Résultats par catégorie

| Catégorie | N | Candidate hit | Hit@1 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| exact_numeric | 5 | 5 | 4 | 5 | 0.850 |
| causal_mechanism | 5 | 3 | 1 | 3 | 0.400 |
| process_description | 4 | 3 | 2 | 3 | 0.625 |
| operator_diagnosis | 4 | 3 | 1 | 2 | 0.333 |
| process_comparison | 2 | 2 | 0 | 2 | 0.500 |
| table_data | 2 | 2 | 2 | 2 | 1.000 |
| impurities_losses_corrosion | 2 | 2 | 2 | 2 | 1.000 |

## Analyse des cinq échecs top-5

Quatre questions n'ont pas leur gold dans les 20 candidats hybrides :

- `Q028` — causal_mechanism
- `Q029` — causal_mechanism
- `Q034` — process_description
- `Q036` — operator_diagnosis

Une question contient son gold dans les candidats, mais celui-ci est retiré du
top-5 par le reranker :

- `Q035` — candidate rank 19, puis absence du top-5 reranké

Tous les autres cas ont une couverture complète de leurs gold dans le top-5.
Pour `Q044`, les trois chunks complémentaires sont présents dans le top-5.

## Conclusion

Le pipeline atteint `Hit@5 = 0.792` et `MRR@5 = 0.628` sur le TEST figé. Les
principales limites observées concernent le rappel candidat des questions
causales et de diagnostic. Ces observations sont rapportées telles quelles :
aucune configuration, requête, pondération, profondeur ou sélection gold n'a
été modifiée après l'ouverture du TEST.

## Empreintes des sorties

- Résumé JSON : `706D50F252A269D4373A2775CEB7E66EF29A1DD1135D66F5EDDDFE232F1AF9DF`
- Détails CSV : `5F6F30C93144A30AB476ECFEF2DB044EE1FBAF07FBA74E85EC5F51337F052F98`
- Manifeste d'exécution : `2F8B25D3B948D2DD597C5CB9A618FD6636AB7318C9F235E39A40EB3EE837850C`
