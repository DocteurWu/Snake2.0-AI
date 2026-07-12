# 🐍 Snake IA 2.0 — Arène Compétitive Multi-Agent (MARL) & Algorithmique

Bienvenue dans **Snake IA 2.0**, une arène de simulation compétitive en temps réel sous forme de **monde persistant synchrone**. Cette arène fait s'affronter **8 architectures de serpents très différentes** (3 agents de Deep Reinforcement Learning PyTorch classiques, 2 agents DQN avancés apprenant par imitation ou de manière collaborative, et 3 agents algorithmiques et mathématiques classiques) dans une grille unique de **40x40** cases contenant par défaut **8 pommes**.

La simulation tourne à l'infini (pas de Game Over global). Lorsqu'un serpent meurt, il subit sa pénalité de collision, son score individuel retombe à 0, et il réapparaît (respawn) instantanément dans une zone libre de l'arène avec sa taille initiale de 3 cases.

---

## 📖 Documentation Scientifique
Pour comprendre la formulation théorique des modèles de Deep Q-Learning (Bellman, target networks) et des architectures alternatives, consultez :
👉 **[Rapport Technique de Modélisation](Rapport_Technique.md)**

---

## 🏗️ Structure Exhaustive du Projet

Le projet est conçu de manière modulaire, dissociant la logique physique, le rendu visuel, la modélisation des réseaux de neurones et l'implémentation des agents décisionnels :

*   **`marl_arena.py`** : Point d'entrée principal. Gère la boucle de rendu avec Pygame (1000x600 px, avec un HUD latéral de 400px dédié aux statistiques de performance), capture les touches utilisateur pour contrôler l'arène en direct et affiche le classement dynamique en temps réel.
*   **`marl_game.py`** : Le moteur physique multi-agent synchrone. Gère l'état de la grille ($40 \times 40$ cellules), le placement des pommes, les déplacements synchrones des serpents, la détection des collisions (murs, corps propres et adverses, face-à-face) et le calcul des récompenses.
*   **`marl_agents.py`** : Implémentation des politiques décisionnelles des 8 serpents. Regroupe les architectures neuronales d'apprentissage profond (DQN), l'algorithme de pathfinding $A^*$, les cartes d'influence physique et les heuristiques de théorie des jeux.
*   **`marl_model.py`** : Définition de la classe PyTorch générique `ReseauDQN`. Construit dynamiquement un réseau de neurones multicouche entièrement connecté (MLP) avec activations ReLU, et intègre des fonctions de sauvegarde/chargement des poids.
*   **`models/`** : Répertoire contenant les fichiers de poids entraînés (`.pth`) et leurs métadonnées (`_meta.pkl`) pour la persistance des performances des agents DQN :
    *   `marl_dqn_standard.pth` (Standard)
    *   `marl_dqn_profond.pth` (Profond)
    *   `marl_dqn_raycast.pth` (Raycast)
    *   `marl_dqn_elite.pth` (Élite)
    *   `marl_dqn_collectif.pth` (Collectif)
*   **`data/`** : Fichiers de données d'apprentissage et historiques :
    *   `benchmark_history.json` : Historique des performances de benchmark (évolution du score moyen sur 2400 épisodes).
    *   `parties_humaines.pkl` : Base de démonstrations humaines enregistrées pour d'éventuels pré-entraînements par imitation ou analyses comparatives.

---

## 🐍 Présentation des 8 Architectures d'Agents

Chaque serpent possède sa propre couleur distinctive et une logique décisionnelle propre :

| ID | Agent | Couleur | Logique & Architecture |
| :--- | :--- | :--- | :--- |
| **0** | **Standard (DQN)** | 🔵 Bleu clair | Réseau PyTorch dense (2 couches : 256/128). Entrée **25D** via une **Grille de Vision Locale de 5x5 cases** située juste devant sa tête et alignée avec sa direction (0: vide, 1: pomme, -1: obstacle). Apprentissage individuel continu. |
| **1** | **Profond (DQN)** | 🔴 Rouge vif | Réseau PyTorch dense plus lourd (4 couches : 256/128/64/32). Entrée **25D** via la grille de vision locale 5x5. Apprentissage individuel continu à haute capacité. |
| **2** | **Raycast (DQN)** | 🟣 Magenta | Réseau PyTorch dense (2 couches : 256/128). Entrée **16D** via 8 rayons/lasers de vision directionnels (à 45°). Chaque rayon renvoie 2 valeurs : distance inverse normée ($1/d$) et identifiant d'objet (`1.0` pomme, `-1.0` corps, `-2.0` mur). |
| **3** | **Mathématicien (A*)** | 🟢 Vert fluo | **Algorithmique pure** : Recherche de chemin optimale de Manhattan ($f(n) = g(n) + h(n)$) vers la pomme la plus proche en traitant tous les corps comme obstacles. *Fallback de survie* : maximisation de l'espace de liberté si encerclé. |
| **4** | **Élite (DQN Sélectif)** | 🟡 Jaune | Réseau PyTorch dense (4 couches : 256/128/64/32) apprenant par **imitation sélective**. Entrée **25D** (vision locale 5x5). Capture et injecte dans son buffer les mouvements de l'expert actuel de l'arène (le leader au classement général). |
| **5** | **Économiste (Influence)**| 🟠 Orange | **Heuristique physique** : Calcule le potentiel global $V(c)$ des 3 cases adjacentes possibles à chaque tick. Attraction des pommes ($+1.0/d$), répulsion forte des corps ($-5.0/d^2$), pénalité doublée pour les têtes adverses proches ($-10.0/d^2$) et répulsion des murs. |
| **6** | **Stratège (GameTheory)** | 💎 Cyan | **Théorie des jeux** : Interception de trajectoire. Repère les adversaires proches (distance $\le 4$), projette leurs futures positions ($t+1$ et $t+2$) et cherche à couper leur route pour provoquer un crash tout en s'orientant vers la nourriture. |
| **7** | **Collectif (DQN Partagé)**| 🌌 Bleu indigo | Réseau PyTorch dense (4 couches : 256/128/64/32) apprenant par **apprentissage centralisé**. Entrée **25D** (vision locale 5x5). À chaque frame, il extrait et mémorise les transitions 25D de **TOUS** les 8 serpents de la grille simultanément. |

---

## 📊 Métrique d'Évaluation : Moyenne Glissante sur 10 Vies

Le classement dynamique affiché dans le HUD est ordonné selon la **moyenne des scores (pommes mangées) des 10 dernières vies** ($M_{10}$) :
$$M_{10}(i) = \frac{1}{|H_i|} \sum_{s \in H_i} s$$
Où $H_i$ est une file circulaire (`deque` de taille max 10) stockant le score historique de chaque serpent lors de ses 10 dernières morts. Si un agent n'est pas encore mort, le score de sa vie en cours fait office de valeur initiale. C'est également ce score glissant qui sert de condition pour la sauvegarde automatique des meilleurs poids des réseaux de neurones.

---

## 🎮 Commandes Interactives & Raccourcis Clavier

L'arène offre un contrôle dynamique en direct grâce à plusieurs raccourcis clavier intégrés :

### ⏱️ Gestion du Temps et de la Vitesse
*   **`[ESPACE]`** : Met en pause ou reprend la simulation.
*   **`[TOUCHE HAUT]`** : Augmente les FPS cibles. Paliers : `5, 10, 15, 25, 40, 60, 120, 250, 500 FPS` et `Illimitée` (vitesse d'entraînement maximale brute sans limite de rafraîchissement).
*   **`[TOUCHE BAS]`** : Diminue les FPS cibles pour observer plus en détail le comportement d'un agent.

### 🍎 Gestion de la Nourriture (Pommes)
*   **`[+]` / `[p]` / `[=]`** : Ajoute une pomme dans l'arène (jusqu'à un maximum de 30 pommes simultanées).
*   **`[-]` / `[o]`** : Retire une pomme de l'arène (jusqu'à un minimum de 1 pomme active).

### ⚙️ Gestion des Statistiques et de la Simulation
*   **`[r]`** : Réinitialise à zéro toutes les statistiques du HUD (morts, score courant, record, pommes cumulées) pour relancer proprement une compétition ou évaluer les modèles sur une nouvelle fenêtre.

### 🐍 Activation / Désactivation Individuelle des Agents
Vous pouvez à tout moment activer ou désactiver chaque agent individuellement à l'aide des touches numériques principales ou du pavé numérique :
*   **`[1]` / `[KP1]`** : Activer / Désactiver le **Standard (DQN)**
*   **`[2]` / `[KP2]`** : Activer / Désactiver le **Profond (DQN)**
*   **`[3]` / `[KP3]`** : Activer / Désactiver le **Raycast (DQN)**
*   **`[4]` / `[KP4]`** : Activer / Désactiver le **Mathématicien (A*)**
*   **`[5]` / `[KP5]`** : Activer / Désactiver l'**Élite (DQN Sélectif)**
*   **`[6]` / `[KP6]`** : Activer / Désactiver l'**Économiste (Influence)**
*   **`[7]` / `[KP7]`** : Activer / Désactiver le **Stratège (GameTheory)**
*   **`[8]` / `[KP8]`** : Activer / Désactiver le **Collectif (DQN Partagé)**

> [!NOTE]
> Lorsqu'un agent est désactivé, son corps disparaît immédiatement de la grille, et ses performances ne sont plus suivies. S'il est réactivé, il réapparaît instantanément dans une zone vacante sous la forme d'un nouveau serpent à taille initiale.

---

## 🚀 Exécution rapide

### 1. Installation des dépendances
Installez les librairies requises :
```bash
pip install torch numpy pygame
```

### 2. Démarrage de l'Arène
```bash
python marl_arena.py
```

### 3. Pré-entraînement par Clonage de Comportement (Imitation Learning)

Pour accélérer l'apprentissage initial des modèles DQN, vous pouvez pré-entraîner de manière supervisée les agents **Standard**, **Profond**, **Élite** et **Collectif** pour imiter les deux meilleurs agents algorithmiques de l'arène (l'Économiste et le Mathématicien A*) :

1. **Générer le Dataset** :
   Exécutez la simulation algorithmique pure pour collecter 200 000 transitions équilibrées (50% Économiste, 50% Mathématicien) :
   ```bash
   python generate_dataset.py
   ```
   *Le dataset sera sauvegardé sous `data/imitation_dataset.pkl`.*

2. **Lancer le Pré-entraînement** :
   Entraînez les réseaux de neurones par classification d'actions (Entropie Croisée) :
   ```bash
   python train_imitation.py
   ```
   *Les poids pré-entraînés seront sauvegardés dans `models/` (ex: `marl_dqn_standard.pth`), avec des métadonnées initialisant l'exploration à 15% ($\epsilon = 0.15$) pour exploiter directement la politique apprise.*

3. **Lancer la Simulation pré-entraînée** :
   Démarrez ensuite l'arène normalement. Les agents DQN commenceront la partie en sachant déjà parfaitement naviguer et chercher de la nourriture :
   ```bash
   python marl_arena.py
   ```

---

## 🧪 État de l'Entraînement Continu (benchmarks)

> **📌 Jusqu'où on est rendus (session 2026-07-10 → 12, Orange Pi 3B)**
>
> On a entraîné les 5 DQN (Standard, Profond, Raycast, Élite, Collectif) en RL continu sur l'arène 8 serpents. Pipeline : dataset imitation 200k (25D + Raycast 16D) → pré-entraînement BC → RL 24h headless avec checkpoints horaires + reprise (`RESUME=1`).
>
> **Optimisations appliquées** : 8 train/tick (×1.5 vitesse vs base), 1 thread torch (le multithread ralentit sur petits réseaux ARM), AMP bf16 désactivé (CPU Rockchip sans bf16).
>
> **Test A+B (reward shaping + curriculum DQN-vs-DQN puis mixte)** : les DQN montent à **Raycast 3.30 en solo** (vs 2.50 max avant), preuve que l'apprentissage est stabilisé. **Mais** dès que les experts réactivent, les DQN retombent à ~1 et se font écraser (Économiste 26, Stratège 13, Mathématicien 17).
>
> **Conclusion** : plafond **structurel** — vision locale 5×5 vs vision globale 40×40 des agents algorithmiques. A+B ne suffisent pas. Pour rivaliser il faudrait l'**option C** (entrée carte d'occupation 40×40 + positions têtes ennemies + CNN), plus lourde mais seul levier qui ferme l'écart.
>
> **Modèles livrés** (`models/`, à jour au 2026-07-12) : Raycast meilleur élève (moy. 8.0), les 4 autres ~6.5. Jouables en solo, dominés en mixte. Benchmarks dans `data/benchmark_curriculum.json`.

---

## 🚀 Entraînement Continu (headless, Orange Pi)

Script d'entraînement 24h reprenable sans perte :
```bash
# Démarrage frais
python run_24h.py

# Reprise après coupure (skip Phase B, recharge poids + compte 24h)
RESUME=1 python run_24h.py
```
Test curriculum A+B (3h) :
```bash
RESUME=1 python run_curriculum.py
```
Logs : `training_24h.log`, `training_curriculum.log`. Checkpoints horaires dans `models/`.
