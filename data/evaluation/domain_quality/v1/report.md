# Évaluation métier domain_quality v1

Ce jeu DEV de production est indépendant des benchmarks historiques. Il
contient 50 questions couvrant les huit ouvrages, les suivis conversationnels
et le français, l’anglais et l’arabe.

Les preuves présentes dans `evidence_draft.jsonl` ne sont pas des golds :
`review_status = needs_human_review`. Les métriques de recall, MRR et précision
des pages ne seront déclarées officielles qu’après revue humaine.

## État validé

- index actif : `kb_quality_20260727_080401` (`quality-v1.2`) ;
- documents actifs : 8 ;
- child chunks : 27 506 ;
- parent chunks : 12 572 ;
- questions DEV : 50, dont 43 françaises, 5 anglaises et 2 arabes ;
- scénarios conversationnels : 13 ;
- smoke validation déterministe : réussie ;
- hard filter automatique : absent ;
- hard filter explicite `/source` : validé ;
- session réelle Qwen/Ollama à huit tours : exécutée ;
- citations syntaxiquement valides : 8/8 tours ;
- langue correcte : 8/8 tours ;
- appels Ollama principaux : 1 par tour ;
- TTFT moyen observé après chargement : 6,624 s ;
- benchmark TEST historique : non lu et non exécuté.

## Limite méthodologique

`retrieval_results.jsonl` et `answer_results.jsonl` restent volontairement
vides tant que les preuves ne sont pas revues par un humain. La validation
actuelle porte sur le schéma du jeu, le routage, la langue, la conversation,
le retrieval réel, l’expansion de contexte et la validité des citations. Elle
ne constitue pas une mesure officielle de Recall@20, Hit@5, MRR@5 ou de
fidélité sémantique.

La configuration B (420/560/70) reste la configuration initiale activée. Elle
n’est pas déclarée gagnante d’une ablation sans gold humain.
