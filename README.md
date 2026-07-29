# PhosProcess Copilot

Assistant RAG local pour les procédés industriels de production d’acide
phosphorique. Le service utilisateur combine la recherche hybride dense +
BM25, le reranker BGE et la sélection `lexical_safeguard_001` du snapshot
DEV figé `dev_best_v3`, puis génère une réponse sourcée avec Qwen via
Ollama.

## Utilisation

Installer et démarrer Ollama, puis rendre le modèle local disponible :

```powershell
ollama serve
ollama pull qwen3:8b
```

Dans l’environnement Python du projet :

```powershell
.\.venv\Scripts\python.exe scripts\ask_phosprocess.py "Pourquoi le lavage du gâteau de gypse améliore-t-il la récupération de P2O5 ?"
```

Pour une conversation interactive avec streaming Ollama réel :

```powershell
.\.venv\Scripts\python.exe scripts\chat_phosprocess.py --show-latency
```

La session accepte `/help`, `/exit`, `/clear`, `/sources`, `/history`,
`/lang auto|fr|en|ar`, `/debug on|off` et :

```text
/source auto|becker|report|thermodynamics|heat_transfer|perry|
        crystallization|control|transport
```

En mode `auto`, le routeur détecte plusieurs domaines et applique uniquement
de faibles bonus documentaires ; aucun ouvrage n’est exclu. Un filtre dur
n’est appliqué qu’après une commande `/source` explicite. Cette détection,
l’expansion terminologique et la résolution normale des suivis sont
déterministes et n’ajoutent aucun appel Ollama.

Pour limiter toute la session au seul document Becker :

```powershell
.\.venv\Scripts\python.exe scripts\chat_phosprocess.py --only-source becker --show-latency
```

La base documentaire de production est administrée séparément des artefacts
d’évaluation. Placez les PDF actifs dans `data/knowledge_base/pdfs/`, puis
exécutez une seule commande :

```powershell
.\.venv\Scripts\python.exe scripts\sync_knowledge_base.py
```

Les options `--dry-run`, `--rebuild`, `--list`, `--status` et `--verbose`
permettent respectivement de prévisualiser, reconstruire, lister, contrôler et
détailler la synchronisation. Une nouvelle version n’est activée qu’après
validation de FAISS, BM25 et du retrieval hybride ; le chat charge la version
indiquée par `data/knowledge_base/current_index.json` à son démarrage.

La reconstruction `kb_quality_*` utilise Docling sans OCR comme parseur
principal, un fallback PyMuPDF contrôlé, des chunks enfant structure-aware,
des parents et des liens `previous`/`next`. L’ancien index reste le rollback
tant que la nouvelle version n’a pas entièrement passé les contrôles.

L’option `--show-query` affiche la question autonome, la langue, le type de
question, les domaines et l’expansion. `--show-retrieval` affiche les cinq
evidence bundles, les parents/voisins ajoutés et leurs scores ;
`--no-history` désactive entièrement la mémoire conversationnelle et
`--no-warmup` désactive le préchauffage unique. La mémoire conserve un résumé
déterministe et les deux derniers échanges, jamais les anciens chunks ou leurs
scores. Les conversations restent uniquement en mémoire et ne sont pas écrites
sur disque.

Pour reproduire le profil de latence sur cinq tours métier :

```powershell
.\.venv\Scripts\python.exe scripts\profile_rag_latency.py
```

Les rapports sans texte documentaire sont écrits dans
`data/observability/latency/`.

Ajouter `--json` pour obtenir toute la réponse structurée, les cinq
sources, leurs métadonnées et leurs scores :

```powershell
.\.venv\Scripts\python.exe scripts\ask_phosprocess.py --json "Comment la température influence-t-elle la cristallisation du gypse ?"
```

Le nom du modèle, l’URL d’Ollama, la température et le timeout sont
configurables dans `configs/rag_production.yaml` ou avec les options de la
CLI. Les paramètres de retrieval restent exclusivement ceux du snapshot
figé `data/evaluation/retrieval/v0.1/frozen/dev_best_v3/`.

## Validation locale

Les tests unitaires n’ont pas besoin d’un serveur Ollama :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
```
