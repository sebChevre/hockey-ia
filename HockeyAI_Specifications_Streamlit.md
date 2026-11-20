# HockeyAI — Spécifications fonctionnelles et techniques du générateur Streamlit

**Version :** 1.0  
**Date :** 20 novembre 2026  
**Statut :** spécification de MVP destinée à une implémentation assistée par IA dans VS Code  
**Contexte :** projet académique de formation continue, présentation le 15 décembre 2026, groupe de 2 à 3 personnes, résultats d'analyse présentés dans des notebooks Jupyter. Budget global indicatif : 40–60 heures, dont environ 12 heures pour le simulateur et son interface.

## 1. Finalité

Construire une application **Streamlit** capable de générer, prévisualiser et exporter des jeux de données synthétiques de hockey sur glace à partir de paramètres configurables. Les données doivent permettre une chaîne de traitement complète (ingestion, contrôle qualité, transformation, exploration), puis au moins deux méthodes d'IA dans des notebooks : **classification supervisée des tirs** et **clustering non supervisé des joueurs**.

**Principe directeur :** Streamlit est uniquement l'interface ; le moteur de simulation est une bibliothèque Python autonome, testable et appelable depuis un notebook ou une CLI. Le simulateur n'entraîne aucun modèle d'IA.

### Objectifs du MVP

1. Générer de 100 à 10 000 matchs reproductibles, avec des équipes et joueurs réutilisés sur plusieurs matchs.
2. Produire des événements horodatés, cohérents et liés (notamment tir → but ou arrêt).
3. Faire varier les distributions sportives via une configuration explicite et sauvegardée.
4. Injecter des anomalies contrôlées **dans une copie** des données de référence.
5. Exporter un ZIP contenant CSV, configuration, métadonnées, dictionnaire et rapport de qualité.
6. Fournir un aperçu visuel et des contrôles de cohérence élémentaires.

### Hors périmètre du MVP

Physique du palet, trajectoires continues, lignes de joueurs et changements en temps réel, règles complètes de hors-jeu/icing, gardien retiré, prolongations et tirs au but, diffusion réseau en direct, base de données serveur, authentification, modèles ML et dashboards ML intégrés.

## 2. Architecture logique

```text
Streamlit (app.py)
  ├── Formulaire de configuration
  ├── Lancement de simulation et progression
  ├── Aperçu des statistiques / qualité
  └── Téléchargement ZIP
             │
             ▼
Python package hockeyai/
  ├── config.py        validation et sérialisation
  ├── roster.py        équipes et joueurs persistants
  ├── engine.py        états et transitions d'événements
  ├── geometry.py      coordonnées, distance, angle
  ├── export.py        fichiers et archive ZIP
  ├── anomalies.py     dégradation contrôlée
  └── validation.py    invariants et rapport
             │
             ▼
Données propres (référence) + copie dégradée + métadonnées
             │
             ▼
Notebooks Jupyter indépendants de Streamlit
```

**Dépendances :** Python 3.11+, streamlit, numpy, pandas, pyarrow, pydantic (ou dataclasses avec validation), plotly, pytest. Préférer les bibliothèques standard pour `zipfile`, `json`, `hashlib` et `pathlib`. Pas de services externes.

## 3. Interface Streamlit

### Page 1 — Générateur

Formulaire avec quatre groupes :

| Paramètre | Type | Défaut | Domaine / règles |
|---|---|---:|---|
| `scenario_name` | Texte | `reference` | 1–50 caractères, nom assaini pour les chemins |
| `n_matches` | Entier | 1000 | 100–10000 |
| `n_teams` | Entier | 12 | 4–16, pair |
| `players_per_team` | Entier | 18 | Fixé à 18 joueurs de champ + 2 gardiens dans le MVP ; affiché en lecture seule |
| `seed` | Entier | 42 | 0–2^32−1 |
| `offensive_intensity` | Entier | 50 | 0–100 ; règle le taux de tirs, pas directement la conversion |
| `team_imbalance` | Entier | 30 | 0–100 ; règle la dispersion des forces des équipes |
| `penalties_per_match` | Flottant | 5.0 | 0–15 ; espérance approximative du total des pénalités |
| `home_advantage` | Flottant | 0.05 | 0–0.20 ; multiplicateur relatif de création d'occasions |
| `anomaly_rate_percent` | Flottant | 2.0 | 0–20 ; part des lignes d'événements ciblées pour corruption |
| `anomaly_types` | Multi-choix | missing, duplicate, invalid_time, unknown_team, invalid_position | Au moins un type si taux > 0 |

Le bouton **Générer** valide les champs, lance la génération et affiche une progression (matchs simulés / total). Bloquer les paramètres pendant le calcul ou figer une copie immuable de la configuration au lancement. Afficher la durée d'exécution, les nombres de matchs, événements, tirs et buts, ainsi que les éventuels avertissements. Ne jamais afficher de succès avant la fin effective de l'exécution.

### Page 2 — Explorer

- Sélection d'un match par `match_id` ; chronologie des événements et score final.
- KPI globaux : matchs, tirs, buts, taux de conversion, pénalités, événements.
- Histogramme des buts par match ; répartition des événements ; nuage de tirs sur la patinoire simplifiée.
- Aperçu tabulaire des cinq datasets, limité par défaut aux 100 premières lignes.
- Rapport d'intégrité sur les **données propres** ; rapport d'anomalies sur la copie dégradée.
- Indiquer clairement que les données sont **synthétiques** et que les graphiques ne sont pas des observations sportives réelles.

### Page 3 — Exporter

- Bouton **Télécharger le ZIP** ; proposer aussi le JSON de configuration seul.
- Afficher le contenu de l'archive et les nombres de lignes.
- La même configuration + seed + version de moteur doit produire des contenus de données identiques (hors métadonnées temporelles). Conserver les tables dans un ordre déterministe.

## 4. Modèle sportif minimal

### 4.1 Terrain et temps

- Match réglementaire : trois périodes de 20 minutes, soit 3600 secondes de jeu.
- `elapsed_seconds` : temps écoulé global dans `[0, 3600]`, en secondes ; `period` : 1, 2 ou 3. Définir explicitement la convention aux frontières des périodes.
- Patinoire fictive de **60 m × 30 m** ; coordonnées `x ∈ [0,60]`, `y ∈ [0,30]`.
- But attaqué défini selon l'équipe et la période ; documenter l'orientation. Pour le calcul des tirs, transformer au besoin les positions dans un repère offensif commun, avec centre du but cible à `(60,15)`.
- Les statistiques spatiales ne prétendent pas reproduire exactement une patinoire réglementaire particulière.

### 4.2 Équipes et joueurs

- Créer `n_teams` équipes dotées de `attack_rating`, `defense_rating` et `goalie_rating` dans `[0,1]`.
- Chaque équipe dispose de 18 joueurs de champ (12 attaquants, 6 défenseurs) et 2 gardiens ; identifiants stables entre matchs.
- Attributs individuels : `shooting_skill`, `passing_skill`, `defensive_skill` (0–1), `position` (`F`, `D`, `G`).
- Les équipes sont appariées de manière déterministe via le générateur aléatoire ; une équipe peut jouer plusieurs matchs.
- Pour le clustering, garantir un nombre suffisant d'observations par joueur et publier `games_played` et `ice_time_seconds` estimé ou simulé ; ne pas utiliser de statistiques par 60 minutes sans dénominateur défini.

### 4.3 Types d'événements

**MVP :** `FACEOFF`, `SHOT`, `SAVE`, `GOAL`, `PENALTY`, `PENALTY_END`. Les tirs ratés peuvent être représentés par `SHOT` avec `shot_outcome=MISSED` sans `SAVE`. Les autres tirs ont `shot_outcome=SAVED` ou `GOAL`.

**Règles obligatoires :**

1. Chaque `GOAL` et `SAVE` référence exactement un `SHOT` via `related_shot_id` ; un tir ne peut avoir qu'une seule issue finale.
2. Le score augmente uniquement à l'occasion d'un `GOAL` ; le score final est recalculable depuis les événements.
3. Les pénalités ont une durée de 120 secondes de jeu dans le MVP, avec expiration à `PENALTY_END` ; la fin anticipée sur but en supériorité peut être omise **si cette simplification est clairement indiquée**.
4. Le moteur maintient un état de situation numérique cohérent (5v5, 5v4, 4v5, éventuellement 4v4) ; ne pas descendre sous trois joueurs de champ par équipe.
5. Un événement de joueur doit référencer un joueur de l'équipe indiquée ; un tir ne peut être attribué à un gardien dans le MVP.
6. L'ordre des événements est déterministe, trié par `(match_id, elapsed_seconds, event_sequence)` ; plusieurs événements peuvent partager la même seconde.
7. Un `FACEOFF` est produit au début de chaque période et après chaque but, sans prétendre modéliser tous les engagements réels.

### 4.4 Génération probabiliste (hybride simplifiée)

Utiliser `numpy.random.Generator(np.random.PCG64(seed))` et éviter les appels à l'état aléatoire global.

Pour chaque match :

1. Initialiser score, période, pénalités actives et équipes.
2. Générer des instants candidats de tirs via un processus de Poisson, avec intensité dépendant de `offensive_intensity`, des forces d'équipes, de l'avantage à domicile et de la situation numérique. Une base indicative est **55 à 70 tirs tentés par match au total**, à calibrer et documenter.
3. Générer des pénalités via un autre processus de Poisson dont l'espérance est `penalties_per_match` ; insérer les débuts et fins de pénalité dans l'ordre temporel.
4. Pour chaque tir, choisir l'équipe selon les intensités relatives, un joueur de champ éligible, une position sur la patinoire et un type de tir.
5. Calculer distance et angle depuis le but visé. Calculer une probabilité de but cachée, bornée, par exemple :

```python
logit_p = (
    beta_0
    + beta_distance * distance_m
    + beta_angle * abs(angle_deg)
    + beta_skill * (shooting_skill - 0.5)
    - beta_goalie * (goalie_skill - 0.5)
    + beta_powerplay * is_powerplay
)
p_goal = clip(sigmoid(logit_p), 0.005, 0.70)
is_goal = rng.binomial(1, p_goal)
```

Les coefficients `beta_*` doivent être **configurés, documentés et calibrés** sur des ordres de grandeur plausibles ; ne pas prétendre qu'ils sont empiriquement estimés. Le modèle de simulation n'est pas un modèle ML entraîné.

6. Si le tir n'est pas un but, tirer une issue `SAVED` ou `MISSED` selon une probabilité documentée ; créer les événements correspondants.
7. Mettre à jour l'état et les scores après chaque événement.

**Important :** sauvegarder `true_goal_probability` dans un fichier de référence réservé à l'évaluation du simulateur, **pas dans le dataset d'entraînement `shots.csv`**. Ne pas inclure `shot_outcome`, le score après le tir ou tout événement futur parmi les features du classificateur.

## 5. Schémas de sortie

Les colonnes suivantes sont obligatoires ; des colonnes supplémentaires sont possibles si elles sont documentées.

### `teams.csv`

`team_id`, `team_name`, `attack_rating`, `defense_rating`, `goalie_rating`.

### `players.csv`

`player_id`, `team_id`, `position`, `shooting_skill`, `passing_skill`, `defensive_skill`.

### `matches.csv`

`match_id`, `home_team_id`, `away_team_id`, `home_score`, `away_score`, `simulation_seed`.

### `events.csv`

`event_id`, `match_id`, `event_sequence`, `elapsed_seconds`, `period`, `event_type`, `team_id`, `player_id`, `x`, `y`, `related_shot_id`, `home_score_before`, `away_score_before`, `manpower_situation`.

Les champs non applicables à un type d'événement peuvent être nuls ; le dictionnaire précise lesquels.

### `shots.csv`

`shot_id`, `event_id`, `match_id`, `team_id`, `player_id`, `elapsed_seconds`, `x`, `y`, `distance_m`, `angle_deg`, `shot_type`, `manpower_situation`, `shooting_skill`, `goalie_skill`, `score_difference_before`, `is_goal`, `shot_outcome`.

`shot_outcome` et `is_goal` sont les **cibles / résultats**, jamais des variables explicatives simultanées.

### `shot_ground_truth.csv` (référence interne, optionnel dans l'export)

`shot_id`, `true_goal_probability`. Destiné à l'audit de la simulation ; ne pas charger dans les notebooks d'entraînement.

### Qualité et métadonnées

- `config.json` : configuration complète et coefficients du modèle de génération.
- `metadata.json` : `generator_version`, `schema_version`, `seed`, date UTC, volumes, conventions de coordonnées et unités, checksum des fichiers.
- `quality_report.json` : comptes d'anomalies demandées, effectivement injectées, détectées par type, invariants et avertissements.
- `data_dictionary.md` : définition de chaque colonne, type, unité, valeurs autorisées, nullabilité et sens métier.

## 6. Injection contrôlée d'anomalies

La génération crée d'abord les fichiers **clean**. Le module d'anomalies copie `events.csv` en `events_corrupted.csv` sans modifier la référence propre.

Types d'anomalies :

- `missing` : retirer une valeur de champ applicable (`team_id`, `player_id` ou `x`).
- `duplicate` : dupliquer une ligne en conservant `event_id` (violation volontaire d'unicité).
- `invalid_time` : rendre `elapsed_seconds` négatif ou supérieur à 3600.
- `unknown_team` : remplacer `team_id` par un identifiant absent de `teams.csv`.
- `invalid_position` : position hors bornes de la patinoire.

Définir `anomaly_rate_percent` comme le **pourcentage de lignes sources sélectionnées** avant corruption, et non le pourcentage final de lignes défectueuses (les doublons augmentent le nombre de lignes). Sélectionner les lignes et types de manière reproductible. Conserver un journal `anomaly_log.csv` avec `source_event_id`, `anomaly_type`, `field`, `original_value`, `corrupted_value` et, pour les doublons, un identifiant technique de ligne injectée. Ce journal est réservé à l'évaluation, pas au nettoyage aveugle.

Le moteur de validation doit vérifier les données propres et signaler séparément les violations introduites dans la copie dégradée.

## 7. Contrats logiciels et interfaces

Implémenter des fonctions pures ou déterministes autant que possible :

```python
from hockeyai.config import SimulationConfig
from hockeyai.engine import simulate_dataset
from hockeyai.validation import validate_dataset
from hockeyai.anomalies import inject_anomalies
from hockeyai.export import build_export_zip

config = SimulationConfig(n_matches=1000, n_teams=12, seed=42)
dataset = simulate_dataset(config)
clean_report = validate_dataset(dataset)
corrupted_events, anomaly_log = inject_anomalies(
    dataset.events, config=config
)
archive_bytes = build_export_zip(
    dataset, corrupted_events, anomaly_log, config, clean_report
)
```

`simulate_dataset` retourne une structure typée contenant les tables `teams`, `players`, `matches`, `events`, `shots` et éventuellement `shot_ground_truth`. Les fonctions ne doivent pas dépendre de Streamlit. L'export doit pouvoir fonctionner sans écriture persistante côté serveur (ZIP en mémoire ou fichiers temporaires nettoyés).

Pour 10 000 matchs, prévoir une progression via callback et limiter les copies de DataFrame. Afficher un avertissement si la mémoire disponible ne permet pas la génération ; ne pas annoncer de garantie de performance non mesurée.

## 8. Critères d'acceptation et tests

| ID | Test | Attendu |
|---|---|---|
| AC-01 | Générer 100 matchs avec seed 42 deux fois | Tables identiques, mêmes tris et mêmes valeurs |
| AC-02 | Changer la seed | Au moins une valeur de simulation diffère |
| AC-03 | Générer 1 000 matchs | 1 000 lignes dans `matches.csv`, aucune exception |
| AC-04 | Vérifier les références | Toutes les FK des données propres existent |
| AC-05 | Vérifier les tirs | Un `SHOT` par ligne `shots.csv` ; chaque `GOAL`/`SAVE` lié à un tir |
| AC-06 | Vérifier les scores | Scores finaux égaux au décompte des `GOAL` par équipe |
| AC-07 | Vérifier le temps | Toutes les secondes sont entre 0 et 3600 et les périodes sont cohérentes |
| AC-08 | Vérifier la géométrie | Positions valides et distances/angles recalculables |
| AC-09 | Injecter 10 % d'anomalies | Journal reproductible ; types sélectionnés présents si volume suffisant |
| AC-10 | Valider les données corrompues | Violations signalées, sans altération des tables propres |
| AC-11 | Exporter le ZIP | Tous les fichiers requis présents, CSV lisibles, JSON valides |
| AC-12 | Lancer hors Streamlit | Génération possible depuis Python ou notebook |
| AC-13 | Contrôler la cible ML | `is_goal` binaire et aucune probabilité cachée dans `shots.csv` |
| AC-14 | Réutilisation des joueurs | Les mêmes `player_id` participent à plusieurs matchs |
| AC-15 | Contrôler les distributions | Rapport des tirs/buts/pénalités par match ; avertir si valeurs hors plages de calibration documentées |

Écrire des tests unitaires pour géométrie, probabilités, configuration et injection d'anomalies ; des tests d'intégration pour 10 et 100 matchs ; un test de charge facultatif pour 10 000 matchs.

## 9. Structure de projet cible

```text
hockeyai-generator/
├── README.md
├── pyproject.toml
├── app.py
├── config/
│   └── example.json
├── hockeyai/
│   ├── __init__.py
│   ├── config.py
│   ├── roster.py
│   ├── engine.py
│   ├── geometry.py
│   ├── anomalies.py
│   ├── validation.py
│   └── export.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_classification.ipynb
│   └── 03_clustering.ipynb
├── tests/
│   ├── test_engine.py
│   ├── test_geometry.py
│   ├── test_anomalies.py
│   └── test_export.py
└── docs/
    └── data_dictionary.md
```

Les notebooks ne font pas partie du développement Streamlit MVP ; seuls les fichiers de données et leurs conventions doivent leur être compatibles.

## 10. Phasage recommandé pour VS Code (≈12 h de développement)

1. **Phase A — Contrats et modèles (2 h)** : créer structure, `SimulationConfig`, schémas, conventions de temps et coordonnées ; tests simples.
2. **Phase B — Moteur déterministe (4 h)** : équipes/joueurs, matchs, tirs/buts/arrêts/pénalités, scores, validation des invariants ; tests.
3. **Phase C — Export et anomalies (3 h)** : CSV, JSON, ZIP, corruption séparée, journal et rapport.
4. **Phase D — Interface Streamlit (2 h)** : formulaire, progression, aperçu, export.
5. **Phase E — Recette (1 h)** : tests de reproductibilité, 100 et 1 000 matchs, vérification de l'archive.

Les estimations sont indicatives. Si le moteur événementiel exige davantage de temps, réduire les visualisations Streamlit plutôt que les contrôles de cohérence et la reproductibilité.

## 11. Instructions à donner à l'assistant IA de VS Code

> Tu es un développeur Python senior. Implémente progressivement le projet décrit dans `HockeyAI_Specifications_Streamlit.md`. Commence par proposer un plan de fichiers et les contrats de données, puis développe la bibliothèque Python indépendante avant l'interface Streamlit. Respecte strictement les invariants sportifs, les clés et les conventions de temps/coordonnées. Utilise `numpy.random.Generator` avec seed pour garantir la reproductibilité. Ajoute des tests pytest pour chaque module. N'invente pas de règles sportives non documentées : consigne les simplifications dans le README. Sépare toujours les données propres, les données corrompues et la vérité cachée du simulateur. Ne crée aucun modèle de Machine Learning dans cette application. Après chaque phase, exécute les tests et indique les résultats et les fichiers modifiés. Ne considère une phase terminée que si ses critères d'acceptation sont vérifiés.

### Ordre conseillé des demandes à l'IA

- **Prompt 1 :** « Réalise uniquement la phase A ; montre-moi les modèles de données, les validations et les tests. »
- **Prompt 2 :** « Réalise la phase B ; génère 10 matchs et prouve les invariants avec pytest. »
- **Prompt 3 :** « Réalise la phase C ; produis une archive exemple et vérifie son contenu. »
- **Prompt 4 :** « Réalise la phase D ; connecte Streamlit aux fonctions existantes sans déplacer de logique métier dans `app.py`. »
- **Prompt 5 :** « Réalise la phase E ; exécute tous les tests, corrige les défauts et documente l'installation et les limites. »

## 12. Définition de terminé

Le MVP est accepté lorsque l'application démarre avec `streamlit run app.py`, génère 1 000 matchs à partir des paramètres par défaut, affiche un aperçu, permet de télécharger une archive exploitable dans Pandas/Jupyter, et que tous les critères AC-01 à AC-14 passent. AC-15 nécessite un rapport de calibration documenté, même si les valeurs cibles restent hypothétiques.

**Limite académique à mentionner explicitement :** les performances futures des modèles ML sur ces données mesureront d'abord leur capacité à retrouver les relations introduites par le simulateur, et ne prouvent pas une validité prédictive sur des matchs réels.
