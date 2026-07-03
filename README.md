# 🐍 Snake IA 2.0 — Arène Compétitive Multi-Agent (MARL) & Algorithmique

Bienvenue dans **Snake IA 2.0**, une arène de simulation compétitive en temps réel sous forme de **monde persistant synchrone**. Cette arène fait s'affronter **8 architectures de serpents très différentes** (3 agents de Deep Reinforcement Learning PyTorch et 5 agents algorithmiques et mathématiques classiques) dans une grille unique de **40x40** cases contenant en permanence **8 pommes**.

La simulation tourne à l'infini (pas de Game Over global). Lorsqu'un serpent meurt, il subit sa pénalité de collision, son score individuel retombe à 0, et il réapparaît (respawn) instantanément dans une zone libre de l'arène avec sa taille initiale de 3 cases.

---

## 📖 Documentation Scientifique
Pour comprendre la formulation théorique des modèles de Deep Q-Learning (Bellman, target networks) et des architectures alternatives, consultez :
👉 **[Rapport Technique de Modélisation](Rapport_Technique.md)**

---

## 🏗️ Architecture du Projet

Le projet est conçu de manière modulaire et propre :
```
Snake2.0-AI/
├── marl_model.py          # Définition PyTorch générique du réseau dense DQN
├── marl_agents.py         # Implémentation des 8 agents (DQN, A*, Minimax, etc.)
├── marl_game.py           # Moteur physique multi-agent synchrone avec détection de collisions et respawn
├── marl_arena.py          # Arène graphique Pygame & HUD du classement dynamique (point d'entrée)
├── data/
│   └── parties_humaines.pkl  # Dataset de parties humaines pour l'agent KNN (Historien)
└── models/
    └── dqn_snake.pth      # Modèle DQN pré-entraîné (optionnel pour Serpent 1)
```

---

## 🐍 Les 8 Architectures d'Agents

Chaque serpent possède sa propre couleur distinctive et une logique décisionnelle propre :

1. **Serpent 1 : Le Standard (Bleu clair)** 
   - **Réseau PyTorch DQN** léger (2 couches cachées, 256 et 128 neurones). 
   - Utilise le vecteur d'état relatif compact (11 dimensions) par rapport à la pomme la plus proche. Apprend en continu via son Replay Buffer à chaque tick.
2. **Serpent 2 : Le Profond (Rouge vif)** 
   - **Réseau PyTorch DQN lourd** (4 couches cachées : 256/128/64/32 neurones).
   - Conçu pour apprendre des stratégies plus complexes avec une plus grande capacité de représentation. Apprend en continu.
3. **Serpent 3 : Le Raycast (Magenta)** 
   - **Réseau PyTorch DQN** avec vision par **lasers/raycast** dans 8 directions (cardinales + diagonales).
   - Reçoit un vecteur de 24 dimensions représentant les distances inverses $1/d$ vers les murs, les pommes et le corps des autres serpents. Apprend en continu.
4. **Serpent 4 : Le Mathématicien (Vert fluo)** 
   - *Sans IA.* Algorithme de recherche de chemin **A* (A-Star)** pur.
   - Calcule à chaque frame le chemin le plus court vers la pomme disponible la plus proche et le suit. Dispose d'un repli de survie si aucun chemin n'est trouvé.
5. **Serpent 5 : Le Psychologue (Jaune)** 
   - *Sans IA.* Algorithme **Minimax avec élagage Alpha-Bêta** (profondeur 3).
   - Simule ses propres coups et ceux du serpent le plus proche pour anticiper ses trajectoires et tenter de bloquer l'adversaire.
6. **Serpent 6 : L'Économiste (Orange)** 
   - *Sans IA.* Algorithme basé sur une **carte thermique d'influence (potentiel)**.
   - Les pommes ont un potentiel attractif (+1), tandis que les murs et tous les corps de serpents à proximité ont un potentiel répulsif (-5). Se déplace vers la case adjacente de potentiel maximal.
7. **Serpent 7 : Le Stratège (Cyan)** 
   - *Sans IA.* Théorie des jeux pure (Matrice de gains géométrique).
   - Analyse la trajectoire projetée des serpents à proximité (distance $\le 4$) et cherche prioritairement à couper leur route pour provoquer des collisions frontales ou latérales ("Kills").
8. **Serpent 8 : L'Historien (Bleu indigo)** 
   - *Sans IA.* Algorithme des **K-Plus Proches Voisins (KNN)** avec $k=5$.
   - Cherche dans `parties_humaines.pkl` les situations les plus similaires à ce qu'il observe et reproduit fidèlement la décision prise par l'humain à l'époque.

---

## 🚀 Lancer la Simulation

### 1. Installation des dépendances
S'assurer que PyTorch, Pygame et Numpy sont installés :
```bash
pip install torch numpy pygame
```

### 2. Exécuter l'arène graphique
```bash
python marl_arena.py
```

### 🎛️ Commandes interactives :
* **[ESPACE]** : Mettre en pause ou reprendre la simulation.
* **[TOUCHE HAUT]** : Augmenter la vitesse de calcul et d'affichage. Paliers : `5, 10, 15, 25, 40, 60, 120, 250, 500 FPS` et `Illimitée` (vitesse maximale brute).
* **[TOUCHE BAS]** : Diminuer la vitesse de simulation.
* **[R]** : Réinitialiser à zéro toutes les statistiques du tableau d'affichage pour relancer une compétition.

---

## 📊 Le Panneau Latéral (HUD)

Le panneau latéral droit affiche en temps réel les indicateurs clés pour comparer l'efficacité des architectures :
* **Classement dynamique** : Trié continuellement par la **moyenne de pommes mangées par vie** (le ratio total de pommes / nombre de morts).
* **Nombre de Respawns** : Le total cumulé de morts et réapparitions de chaque agent.
* **Record de longueur** : La longueur maximale absolue atteinte par chaque serpent depuis le début de la session.
* **Taux d'exploration ($\epsilon$)** : Affiche en direct le facteur d'exploration décroissant de chaque DQN.
