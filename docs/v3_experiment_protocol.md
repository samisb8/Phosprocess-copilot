# Protocole expérimental v3

## Statut de la v2

La v2 est la baseline officielle et immuable. Les artefacts suivants ne doivent
plus être modifiés ni régénérés :

- le gold TEST v2 et son snapshot figé ;
- les composants du snapshot `dev_best_v2` ;
- les résultats de l'unique run TEST v2 ;
- les métriques officielles et le rapport final v2.

Le TEST v2 peut uniquement être cité dans une comparaison rétrospective. Il ne
doit fournir ni requête cible, ni vocabulaire d'expansion, ni règle de scoring,
ni seuil, ni choix de modèle pour la v3.

## Développement de la v3

Toutes les hypothèses, implémentations et décisions de sélection de modèle v3
sont évaluées exclusivement sur DEV Q001-Q016. Le point de départ reproductible
est le snapshot en lecture seule :

`data/evaluation/retrieval/v0.1/frozen/dev_best_v2`

Les nouveaux artefacts sont placés sous des chemins v3 séparés. Un script v3 ne
doit jamais utiliser `test_pool_v2`, `test_gold_final_v2` ou les sorties
`test_hybrid_reranked_v2_*` comme entrée.

L'analyse initiale est produite avec :

```powershell
python scripts/analyze_retrieval_dev_v3.py
```

Le script contrôle les empreintes du snapshot DEV, exige exactement Q001-Q016,
recalcule les métriques et écrit uniquement dans :

`data/evaluation/retrieval/v0.1/v3/analysis`

## Discipline expérimentale

Chaque expérience v3 doit enregistrer :

1. l'hypothèse testée ;
2. les fichiers et empreintes d'entrée DEV ;
3. la configuration complète ;
4. les métriques par requête et agrégées ;
5. la règle de sélection annoncée avant comparaison ;
6. le statut retenu ou rejeté, sans regarder le TEST v2.

La v3 finale est choisie et figée sur DEV. Son évaluation finale exige ensuite
un nouveau jeu de test indépendant, sans réutiliser Q021-Q048 comme nouvelle
mesure finale. Ce nouveau test est exécuté une seule fois après gel de la
configuration v3 et produit un rapport distinct de la baseline v2.
