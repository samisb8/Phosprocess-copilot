# Robustesse DEV du candidat v3 lexical_safeguard_001

## Périmètre

- Split utilisé : DEV uniquement.
- Répétitions complètes : 3.
- Artefact TEST lu ou exécuté : non.
- Gold DEV utilisé dans l'inférence : non ; métriques uniquement.
- Gel automatique de v3 : non.

## Audit de la politique

- Audit statique réussi : true.
- Aucun query_id, chunk précis, gold ou texte de réponse codé en dur.
- Entrées du safeguard : candidats, rangs BM25/hybrides et ordre du reranker.

## Déterminisme

- Top-5 courant identique sur les 3 runs : true.
- Métriques courantes identiques : true.
- Toutes les sélections de toutes les variantes sont stables : true.

## Sensibilité du paramètre lexical_slots

| Variante | slots | Candidate Recall@20 | Evidence Recall@5 | Hit@5 | MRR@5 | Hit@1 | Rang |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_lexical_slots_0 | 0 | 1.0000 | 0.9375 | 0.9375 | 0.7760 | 0.6875 | 2 |
| lexical_safeguard_001 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.7885 | 0.6875 | 1 |
| permissive_lexical_slots_2 | 2 | 1.0000 | 0.9375 | 0.9375 | 0.7760 | 0.6875 | 3 |

## Différences par question

| Variante | Question | Rang v2 | Rang variante | Résultat |
|---|---|---:|---:|---|
| lexical_safeguard_001 | Q003 | 1 | 1 | selection_changed_metric_equal |
| lexical_safeguard_001 | Q006 | 4 | 4 | selection_changed_metric_equal |
| lexical_safeguard_001 | Q008 | 1 | 1 | selection_changed_metric_equal |
| lexical_safeguard_001 | Q009 | 1 | 1 | selection_changed_metric_equal |
| lexical_safeguard_001 | Q010 | 1 | 1 | selection_changed_metric_equal |
| lexical_safeguard_001 | Q011 | 3 | 3 | selection_changed_metric_equal |
| lexical_safeguard_001 | Q012 | 1 | 1 | selection_changed_metric_equal |
| lexical_safeguard_001 | Q015 | MISS | 5 | improved |
| permissive_lexical_slots_2 | Q002 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q003 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q004 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q005 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q006 | 4 | MISS | regressed |
| permissive_lexical_slots_2 | Q008 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q009 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q010 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q011 | 3 | 3 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q012 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q013 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q014 | 1 | 1 | selection_changed_metric_equal |
| permissive_lexical_slots_2 | Q015 | MISS | 4 | improved |
| permissive_lexical_slots_2 | Q016 | 1 | 1 | selection_changed_metric_equal |

## Recommandation

**freeze lexical_safeguard_001**

Cette recommandation est fondée uniquement sur DEV. Aucun snapshot dev_best_v3 n'a été créé.
