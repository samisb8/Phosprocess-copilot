# Audit technique — PhosProcess Copilot RAG v1

**État audité :** branche `fix-rag-grounding`, après le Patch 3.9.2 et les smoke tests fonctionnels du 29 juillet 2026.

## 1. Verdict exécutif

PhosProcess Copilot utilise maintenant un **RAG local hybride, hiérarchique, conversationnel, orienté rôles de preuve et contrôlé par contrats de réponse**.

Ce n’est pas un RAG simple de type « embedding → top-k → prompt ». La chaîne active comprend :

1. classification de la question et résolution des suivis ;
2. routage domaine/source avec filtre dur uniquement lorsque la source est explicitement imposée ;
3. planification de plusieurs rôles de preuve ;
4. recherche hiérarchique par sections ;
5. récupération dense BGE-M3 + sparse BGE-M3 + BM25 ;
6. fusion RRF ;
7. scoring ColBERT dynamique avec BGE-M3 ;
8. reranking cross-encoder BGE ;
9. sélection sémantique des preuves par rôle ;
10. expansion parent/voisins ;
11. génération locale par Qwen3:8B via Ollama ;
12. validation des citations ;
13. suppression des affirmations non prouvées ;
14. contrats déterministes pour les réponses sensibles ;
15. liaison finale phrase → rôle → chunk → numéro de source.

**Statut :** baseline RAG v1 fonctionnelle et validée. Elle est prête à devenir le cœur d’une API, mais le produit n’est pas encore déployé ni instrumenté comme un service de production.

---

## 2. Nature exacte de l’architecture

### Type de système

- **Local** : le générateur est exécuté par Ollama.
- **Multi-document** : huit documents actifs dans le catalogue.
- **Hybride** : dense, sparse et lexical.
- **Hiérarchique** : sélection de sections avant les chunks finaux.
- **Parent-enfant** : chunks enfants pour la précision, parents/voisins pour le contexte.
- **Role-aware** : chaque type de question exige des rôles de preuve précis.
- **Contract-driven** : certaines réponses sont reconstruites de façon déterministe lorsque la sortie libre du LLM n’est pas assez fiable.
- **Citation-grounded** : une phrase numérique ou technique doit être soutenue par le chunk cité.
- **Conversationnel** : suivi des entités, de la source verrouillée et des deux derniers échanges.

### Ce que le système n’est pas encore

- pas une API FastAPI ;
- pas un système agentique ;
- pas un modèle fine-tuné ;
- pas un service Docker de production ;
- pas encore doté d’un tracing LLMOps complet ;
- pas un système multi-utilisateur persistant.

---

## 3. Architecture globale

```text
                         MODE HORS LIGNE

PDF actifs
   │
   ▼
Catalogue documentaire + contrôles SHA-256
   │
   ▼
Docling ──fallback contrôlé──> PyMuPDF
   │
   ▼
Nettoyage + extraction structurée
   │
   ▼
Chunking hybride parent/enfant
(chapitres, sections, tableaux, équations, figures)
   │
   ▼
Post-traitement + validation
   │
   ├──> Index dense FAISS / BGE-M3, 1024 dimensions
   ├──> Index sparse BGE-M3
   ├──> Index BM25
   └──> Index hiérarchique de sections
   │
   ▼
Version kb_quality_* validée puis activée atomiquement


                         MODE EN LIGNE

Question utilisateur
   │
   ▼
Validation + langue + classification + mémoire
   │
   ▼
Résolution du follow-up et de l’entité
   │
   ▼
Routage domaine/source
   │
   ▼
Plan de rôles de preuve
   │
   ├──> recherche hiérarchique par sections
   ├──> dense BGE-M3
   ├──> sparse BGE-M3
   └──> BM25
          │
          ▼
       Fusion RRF
          │
          ▼
   ColBERT dynamique BGE-M3
          │
          ▼
   BGE reranker v2-m3
          │
          ▼
Validation sémantique des rôles
          │
          ▼
5 evidence bundles + parent/voisins
          │
          ▼
Prompt contrôlé
          │
          ▼
Qwen3:8B via Ollama
          │
          ▼
Validation citationnelle
          │
          ├── rejet / pruning des affirmations non soutenues
          └── contrat déterministe selon le type de question
          │
          ▼
Réponse finale + sources exactes + métriques de latence
```

---

## 4. Chemin d’exécution principal et fichiers

### 4.1 Entrée utilisateur

| Responsabilité | Fichier principal |
|---|---|
| Démarrer le chat terminal | `scripts/chat_phosprocess.py` |
| Boucle interactive, affichage des sources et latences | `src/phosprocess/rag/terminal_chat.py` |
| Orchestration complète d’un tour RAG | `src/phosprocess/rag/pipeline.py` |
| Client local Qwen/Ollama et streaming | `src/phosprocess/llm/ollama_client.py` |
| Configuration runtime du modèle et de la mémoire | `configs/rag_production.yaml` |

Le point central est `PhosProcessRAG.stream_answer()` dans `rag/pipeline.py`.

### 4.2 Compréhension de la question

| Fonction | Fichier |
|---|---|
| Validation de longueur et contenu | `rag/pipeline.py` → `validate_question()` |
| Détection de langue | `rag/language.py` |
| Classification : définition, explication, bilan, procédé, etc. | `rag/question_classifier.py` |
| Détection d’une requête directe sans RAG | `rag/adaptive_router.py` |
| Résolution des pronoms et follow-ups | `rag/followup_resolver.py` |
| État métier : entité, source active, verrou | `rag/conversation_state.py` |
| Résumé et fenêtre conversationnelle | `rag/conversation_memory.py` |

La mémoire ne conserve pas les anciens chunks ni leurs scores. Elle conserve un résumé déterministe, les deux derniers échanges et l’état métier.

### 4.3 Routage documentaire

| Fonction | Fichier |
|---|---|
| Détecter Becker, Bird, rapport OCP, Perry, etc. | `retrieval/domain_router.py` |
| Définir les domaines et petits boosts documentaires | `retrieval/domain_router.py` |
| Gérer les modes de source et labels | `rag/source_policy.py` |
| Catalogue des huit documents | `configs/knowledge_base_catalog.yaml` |

Règle importante :

- mode automatique → tous les documents restent accessibles, avec de faibles boosts ;
- source explicitement demandée → filtre dur sur ce document ;
- follow-up → le verrou explicite est conservé ;
- nouvelle question indépendante → retour au mode automatique, sauf nouvelle source explicite.

### 4.4 Planification des preuves

| Fonction | Fichier |
|---|---|
| Décomposer la question en rôles atomiques | `retrieval/retrieval_planner.py` |
| Définir `EvidenceRole` et `RetrievalPlan` | `retrieval/retrieval_planner.py` |
| Expansion terminologique technique | `retrieval/query_expansion.py` |

Exemples :

```text
Question bilan P2O5
→ p2o5_conservation
→ p2o5_feed
→ p2o5_product
→ p2o5_entrainment
```

```text
Question Bird
→ momentum_transport
→ velocity_gradient
→ newton_viscosity_law
```

```text
Question pompe Becker
→ pump_circulation / pump_withdrawal
→ pump_heating_path
→ pump_process_function
```

### 4.5 Retrieval v4

| Composant | Fichier |
|---|---|
| Orchestrateur du retriever qualité | `rag/quality_retrieval.py` |
| Recherche planifiée multi-rôles | `retrieval/quality_hybrid.py` |
| Dense BGE-M3 + FAISS | `retrieval/dense.py` |
| Sparse BGE-M3 persistant | `retrieval/bge_sparse.py` |
| BM25 technique | `retrieval/bm25.py` |
| Structures et configuration hybride/RRF | `retrieval/hybrid.py` |
| Recherche hiérarchique par sections | `retrieval/hierarchical.py` |
| BGE reranker cross-encoder | `reranking/reranker.py` |
| Sélection figée `lexical_safeguard` | `retrieval/v3_selection.py` |

#### Fusion utilisée

La première étape récupère chaque rôle séparément avec :

```text
Dense BGE-M3
+ BGE-M3 sparse
+ BM25
```

Les rangs sont fusionnés par **RRF**. Une petite contribution du score ColBERT est ensuite ajoutée avant le cross-encoder.

Le retriever réserve aussi plusieurs candidats forts pour chaque rôle afin qu’une preuve rare, mais exacte, ne soit pas éliminée par un passage général mieux classé globalement.

### 4.6 Sélection et expansion des preuves

| Fonction | Fichier |
|---|---|
| Vérifier qu’un chunk soutient réellement un rôle | `retrieval/evidence_roles.py` |
| Rejeter sommaires, listes de figures et faux positifs | `retrieval/evidence_roles.py` |
| Vérifier les preuves obligatoires d’un type de question | `retrieval/evidence_coverage.py` |
| Construire les bundles de preuve | `retrieval/evidence_bundle.py` |
| Ajouter parent et voisins | `retrieval/context_expander.py` |
| Préparer une fenêtre de contexte | `rag/context_window.py` |

Le Patch 3.9.2 a renforcé cette étape : un tag `evidence_role` ne suffit plus. Le texte du chunk doit réellement contenir les concepts, nombres et unités requis.

### 4.7 Prompt et génération

| Fonction | Fichier |
|---|---|
| Prompt par type de question | `rag/prompts.py` |
| Limites de contexte et réponse | `configs/rag_production.yaml` |
| Qwen3:8B, température, timeout | `configs/rag_production.yaml` |
| Appel Ollama, streaming et erreurs | `llm/ollama_client.py` |

Configuration principale actuelle :

```text
Modèle             qwen3:8b
Température        0.1
Contexte Ollama    8192 tokens
Sortie maximale    450 tokens
Contexte documentaire total 2600 tokens
Evidence bundles   5
```

### 4.8 Validation après génération

| Fonction | Fichier |
|---|---|
| Extraire et valider les numéros de source | `rag/citations.py` |
| Mesurer le soutien d’une affirmation | `rag/fidelity.py` |
| Supprimer les affirmations non prouvées | `rag/fidelity.py` → `prune_unsupported_claims()` |
| Réponses déterministes | `rag/fidelity.py` |
| Contrats par type de réponse | `rag/fidelity.py` → `enforce_answer_contract()` |

Séquence réelle :

```text
Sortie libre de Qwen
→ validation des citations
→ si échec : pruning déterministe
→ contrat de réponse
→ nouvelle validation
→ réponse finale ou insuffisance contrôlée
```

Pour les cas sensibles, le système ne fait pas confiance aveuglément au texte du LLM. Il reconstruit une réponse avec les bundles validés.

---

## 5. Architecture d’ingestion et d’indexation

### Synchronisation de la base

| Étape | Fichier |
|---|---|
| Commande principale de synchronisation | `scripts/sync_knowledge_base.py` |
| Gestion des versions `kb_quality_*` | `knowledge_base/quality_manager.py` |
| Chargement de la version active | `knowledge_base/runtime.py` |
| Manifeste, SHA et métadonnées | `knowledge_base/manifest.py` |
| Catalogue documentaire | `knowledge_base/catalog.py` |
| Construction qualité | `knowledge_base/quality_indexing.py` |
| Corpus qualité | `knowledge_base/quality_corpus.py` |

### Extraction et chunking

| Étape | Fichier |
|---|---|
| Extraction Docling | `ingestion/docling_extractor.py` |
| Fallback PDF | `ingestion/pdf_fallback.py` et `parser_router.py` |
| Validation d’extraction | `ingestion/extraction_quality.py` |
| Chunking technique | `ingestion/technical_chunker.py` |
| Sérialisation | `ingestion/chunk_serialization.py` |
| Validation de chunks | `ingestion/chunk_validation.py` |
| Nettoyage/post-traitement | `preprocessing/cleaner.py`, `chunk_postprocessor.py` |

### Pipelines reproductibles

```text
pipelines/ingest_documents.py
pipelines/clean_pages.py
pipelines/build_chunks.py
pipelines/postprocess_chunks.py
pipelines/validate_ingestion.py
pipelines/validate_chunks.py
pipelines/build_dense_index.py
pipelines/validate_dense_index.py
pipelines/build_bm25_index.py
```

Le runtime n’exécute pas ces scripts à chaque question. Ils servent à construire une nouvelle version de la base. Une fois les index validés, le chat lit la version active.

---

## 6. Configuration : quels fichiers sont réellement importants ?

### Runtime actif

```text
configs/rag_production.yaml
configs/quality_pipeline.yaml
configs/knowledge_base_catalog.yaml
```

### Modèles et construction des index

```text
configs/embeddings.yaml
configs/reranking.yaml
configs/retrieval.yaml
configs/retrieval_v2.yaml
configs/chunking.yaml
configs/chunk_postprocessing.yaml
```

### Baseline figée d’évaluation

Le runtime charge les paramètres de sélection depuis :

```text
data/evaluation/retrieval/v0.1/frozen/dev_best_v3/
```

Fichiers principaux :

```text
freeze_manifest.json
lexical_safeguard_v3.yaml
retrieval_v2.yaml
reranking.yaml
sha256.csv
```

La fonction `load_frozen_v3_config()` dans `rag/pipeline.py` vérifie l’intégrité du snapshot. Le Patch 3.8.1 a séparé l’intégrité du snapshot historique et le code runtime v4 afin d’éviter de bloquer l’évolution du retriever.

---

## 7. Historique des changements par familles de patches

### 3.2 à 3.7.1 — Grounding et couverture de procédé

Objectif : passer d’un RAG qui récupère des passages plausibles à un RAG qui couvre les étapes et flux nécessaires.

Principales évolutions :

- diagnostics de retrieval ;
- couverture entrée/sortie des flux ;
- restauration du chemin de procédé ;
- récupération adaptative des preuves manquantes ;
- planification atomique des preuves ;
- normalisation de l’ordre des étapes.

### 3.8 à 3.8.2 — Retriever v4

Objectif : renforcer la récupération avant la génération.

- recherche multi-rôles ;
- dense + sparse + BM25 ;
- ColBERT dynamique ;
- fusion globale et réservations par rôle ;
- séparation du snapshot figé et du runtime ;
- contrôle d’alignement exact de l’index sparse.

### 3.8.3 à 3.8.6.1 — Contrats de preuve et réponses déterministes

Objectif : empêcher qu’une réponse correcte en apparence soit soutenue par les mauvaises sources.

- rôles de preuve ;
- garde de périmètre documentaire ;
- contrats de réponse de bout en bout ;
- builders déterministes ;
- alignement des rôles du process flow ;
- provenance rôle → chunk → citation.

### 3.8.7 à 3.9.0.2 — Conversation, source et routage

Objectif : rendre le RAG conversationnel sans perdre le contrôle documentaire.

- classification d’intention ;
- mémoire de l’entité ;
- source explicite persistante ;
- corpus spécialisés Becker, rapport OCP, Bird, etc. ;
- routage correct de la diffusion de quantité de mouvement vers la mécanique des fluides.

Le correctif 3.9.0.1 n’appartient pas à la baseline finale : il a été remplacé par l’installateur 3.9.0.2 validé.

### 3.9.1 — Evidence-to-Claim Integrity Closure

Objectif : exiger que le chunk cité contienne réellement la valeur ou le mécanisme affirmé.

- conservation de la source après erreur ;
- contrat de pompe Becker ;
- validation exacte de 18,03 t/h, 18 t/h et 30 kg/h ;
- rejet des passages Fick pour la loi de Newton.

### 3.9.2 — Composite Balance & Exact Citation Binding

Objectif : fermer la dernière rupture entre preuves atomiques et phrase finale.

- bilan P2O5 construit à partir de trois preuves distinctes ;
- validation sémantique des rôles ;
- rejet des chunks corrosion/électricité comme preuve du trajet hydraulique ;
- citations exactes des définitions ;
- liaison phrase → rôle → chunk_id → source_number.

---

## 8. Exemple de trace complète

### Question

```text
Établis le bilan de P2O5 de l’échelon J de JFC4 selon le rapport OCP.
```

### Fichiers traversés

```text
scripts/chat_phosprocess.py
  ↓
rag/terminal_chat.py
  ↓
rag/pipeline.py
  ↓
rag/question_classifier.py          type=balance
  ↓
retrieval/domain_router.py          source=rapport OCP
  ↓
retrieval/retrieval_planner.py      4 rôles P2O5
  ↓
rag/quality_retrieval.py
  ↓
retrieval/quality_hybrid.py         dense+sparse+BM25+ColBERT
  ↓
reranking/reranker.py
  ↓
retrieval/evidence_roles.py         validation valeur/unité
  ↓
retrieval/context_expander.py       bundles
  ↓
rag/prompts.py
  ↓
llm/ollama_client.py                Qwen3:8B
  ↓
rag/citations.py
  ↓
rag/fidelity.py                     bilan composite déterministe
  ↓
rag/terminal_chat.py                affichage final
```

### Résultat attendu

```text
18,03 t/h alimentation  → chunk page 36
18,00 t/h produit       → chunk pages 38–39
30 kg/h entraînement    → chunk page 40
```

Aucune phrase numérique n’est produite sans preuve contenant la valeur correspondante.

---

## 9. Tests et validation

### Tests automatisés

Répertoire :

```text
tests/
```

Tests les plus importants pour l’architecture actuelle :

```text
tests/test_retrieval_planner_v4.py
tests/test_evidence_roles_v4.py
tests/test_evidence_coverage.py
tests/test_answer_contracts_v4.py
tests/test_quality_retrieval.py
tests/test_quality_routing_and_expansion.py
tests/test_language_classifier_followup.py
tests/test_source_policy.py
tests/test_terminal_chat.py
```

### Benchmark et jeux d’évaluation

```text
artifacts/retrieval_v4_benchmark.json
data/evaluation/domain_quality/
data/evaluation/retrieval/v0.1/
```

### Smoke tests validés

- source Becker persistante sur les follow-ups ;
- réponse pompe reliée aux bonnes pages ;
- bilan P2O5 exact et sans fallback ;
- Bird sans citation de la diffusion de masse ;
- refus d’un salaire absent ;
- refus d’inventer un diamètre de pompe.

---

## 10. Forces actuelles

1. **Grounding fort** : la réponse finale est contrôlée après génération.
2. **Hybridation complète** : dense, sparse, lexical et reranking.
3. **Gestion des preuves rares** : réservation de candidats par rôle.
4. **Routage documentaire explicable** : source, domaine et filtre sont loggés.
5. **Conversation contrôlée** : la mémoire n’injecte pas d’anciens chunks.
6. **Versionnement de la base** : nouvelle version activée seulement après validation.
7. **Reproductibilité** : snapshot DEV figé et manifestes SHA.
8. **Refus correct** : absence de preuve → pas d’invention.

---

## 11. Faiblesses et dette technique

### Priorité haute avant un vrai déploiement

1. `rag/pipeline.py` est un **orchestrateur monolithique** de plus de 3 000 lignes.
2. `rag/fidelity.py` concentre trop de responsabilités : parsing, support sémantique, templates et contrats.
3. Il n’existe pas encore d’API stable ni de schéma de requête/réponse public.
4. L’observabilité actuelle mesure surtout la latence ; elle ne fournit pas encore un tracing distribué complet.
5. Les métriques de production, feedback utilisateur et dérive documentaire ne sont pas encore persistés.
6. La gestion multi-utilisateur et l’isolation des sessions ne sont pas encore traitées.

### Priorité moyenne

1. Des résultats secondaires bruités peuvent encore apparaître comme `reranker_fill` ; ils sont filtrés avant la réponse, mais augmentent le coût et le bruit des logs.
2. Les contrats déterministes sont très solides pour les cas couverts, mais doivent être étendus via un benchmark plus large.
3. La génération locale reste lente : environ 27–35 s sur les questions documentées des smoke tests.
4. Les fichiers historiques de sauvegarde présents dans `src/` comme `*.broken` ou `*.before_*` devraient être retirés du code suivi s’ils existent encore dans le dépôt.
5. Les fichiers Docker/MLOps initiaux ne constituent pas encore un déploiement validé.

---

## 12. Prochaine étape recommandée

Ne pas modifier le retriever ni reconstruire les index maintenant.

Avant FastAPI, réaliser un petit **refactoring sans changement fonctionnel** :

```text
rag/pipeline.py
  ├── conversation_orchestrator.py
  ├── retrieval_service.py
  ├── generation_service.py
  └── answer_validation_service.py

rag/fidelity.py
  ├── claim_support.py
  ├── deterministic_builders.py
  ├── answer_contracts.py
  └── citation_binding.py
```

Ce refactoring doit être couvert par les tests existants et ne doit changer ni les réponses, ni les index, ni le benchmark.

Ensuite seulement :

```text
FastAPI
→ Docker Compose
→ tracing/MLflow/OpenTelemetry
→ CI/CD
→ staging
→ collecte de données
→ fine-tuning
→ outils et agentic RAG
```

---

## 13. Conclusion

La baseline actuelle est une architecture RAG avancée et cohérente :

```text
RAG hybride
+ recherche hiérarchique
+ evidence planning
+ mémoire contrôlée
+ routage documentaire
+ Qwen local
+ contrats de réponse
+ citation binding
```

Le travail des patches n’a pas seulement « corrigé quelques bugs ». Il a progressivement transformé un pipeline de recherche classique en un système **evidence-first**, où la génération est subordonnée aux preuves récupérées.

La prochaine phase n’est plus de reconstruire le RAG. Elle consiste à **stabiliser son organisation logicielle**, l’exposer par API, le conteneuriser et l’observer en production.
