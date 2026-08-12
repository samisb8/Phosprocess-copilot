# Environnement Docker local

La stack de développement conserve ses ports hôte par défaut et démarre avec :

```powershell
docker compose --env-file .env.compose up -d
```

Pour préparer le staging, copier `.env.staging.example` vers `.env.staging`, puis
remplacer les valeurs sensibles localement. Le fichier `.env.staging` est ignoré
par Git. Démarrer ensuite le staging avec :

```powershell
docker compose --env-file .env.staging up -d
```

La valeur `COMPOSE_PROJECT_NAME=phosprocess-staging` place les conteneurs, le
réseau et les volumes Compose (`postgres_data`, `prometheus_data` et
`grafana_data`) dans un namespace distinct de celui du développement. Les ports
hôte staging sont également différents, tandis que les ports internes restent
inchangés.

Pour arrêter le staging sans supprimer ses données :

```powershell
docker compose --env-file .env.staging stop
```

Ne pas utiliser `down -v`, car cette commande supprimerait les volumes et leurs
données persistantes.
