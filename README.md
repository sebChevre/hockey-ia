# HockeyAI Generator

Application Streamlit de génération de données synthétiques de hockey. Les
données sont artificielles et ne représentent pas des observations sportives
réelles; l’application n’entraîne aucun modèle d’IA.

## Installation et lancement

Depuis la racine du projet :

```bash
uv sync --extra test
uv run streamlit run app.py
```

Le parcours est divisé en trois pages :

- **Générateur** : configurer les matchs, lancer la simulation et contrôler sa
  progression.
- **Explorer** : consulter les statistiques, la chronologie d’un match, les
  tirs sur la patinoire simplifiée et les rapports d’intégrité.
- **Exporter** : télécharger le ZIP complet ou la configuration JSON seule.

Les données sont gardées dans la session Streamlit en cours; elles ne sont pas
persistées côté serveur. Les paramètres et événements sont traités par la
bibliothèque `hockeyai`, qui reste utilisable indépendamment de Streamlit.
La page Générateur affiche une estimation mémoire indicative lorsque le volume
demandé dépasse la mémoire disponible; cette heuristique n’est pas une garantie
de performance ou de réussite.

## Limites du modèle

Le moteur est une simulation événementielle simplifiée, pas une estimation
empirique du hockey réel. Les coefficients de probabilité de but, les
distributions de tirs, les profils d’équipe et les plages de calibration sont
des hypothèses synthétiques documentées dans
[`docs/simulation_conventions.md`](docs/simulation_conventions.md). Une bonne
performance future d’un modèle ML sur ces données mesurera d’abord sa capacité
à retrouver les relations introduites par le simulateur; elle ne prouvera pas
sa validité prédictive sur des matchs réels.

## Tests

```bash
uv run pytest
```
