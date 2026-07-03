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
   - **Mécanique** : Recherche de chemin **A* (A-Star)** pure.
   - **Comment ça marche** : À chaque tick, il calcule la distance de Manhattan par rapport à toutes les pommes actives et sélectionne la plus proche. Il génère ensuite un graphe de navigation temporaire où les cases occupées (par son corps ou les corps de tous les autres serpents) sont marquées comme obstacles. Il exécute A* pour trouver le chemin optimal de sa tête vers cette pomme.
   - **Repli (Survival Fallback)** : Si A* ne trouve aucun chemin (serpent encerclé ou bloqué), il calcule le nombre de voisins vides autour de ses 3 coups adjacents possibles et choisit la direction qui offre le plus grand espace de liberté pour retarder la collision.
5. **Serpent 5 : Le Psychologue (Jaune)** 
   - **Mécanique** : Algorithme **Minimax** de théorie des jeux avec élagage **Alpha-Bêta** (profondeur 3).
   - **Comment ça marche** : Il repère le serpent adverse dont la tête est la plus proche de la sienne. Il modélise la situation comme un jeu à deux joueurs à somme nulle : lui-même (Max) cherche à maximiser son score de survie et d'accès aux pommes, et l'adversaire (Min) cherche à minimiser ce score en le bloquant. L'arbre simule l'action de Max, la réaction de Min, puis le coup suivant de Max.
   - **Fonction d'évaluation** : Évalue chaque feuille de simulation en récompensant la survie, la proximité avec la pomme la plus proche, et en pénalisant fortement la collision. Si la tête du Psychologue arrive à 1 case de la cellule projetée devant l'adversaire (interception de trajectoire), un fort bonus de blocage lui est attribué.
6. **Serpent 6 : L'Économiste (Orange)** 
   - **Mécanique** : Calcul tridimensionnel sur **carte d'influence (Heatmap / Champ de potentiel)**.
   - **Comment ça marche** : Sans planifier sur plusieurs coups, il applique une analyse physique de champ de forces locaux sur ses 3 mouvements possibles (devant, gauche, droite). 
   - **Calcul du potentiel** : Pour chaque case cible candidate :
     - Les pommes agissent comme des charges positives attirant le serpent : $+1.0 / (\text{distance} + 0.5)$.
     - Les segments de corps (soi et adversaires) agissent comme des charges négatives répulsives : $-5.0 / (\text{distance} + 0.5)$.
     - Les têtes adverses à proximité immédiate doublent cette répulsion ($-10.0$) pour éviter les collisions frontales.
     - Les murs de la grille exercent également une force de répulsion pour éviter que le serpent ne se coince contre les bords.
     - L'agent choisit simplement le mouvement qui mène à la cellule ayant le potentiel total le plus élevé.
7. **Serpent 7 : Le Stratège (Cyan)** 
   - **Mécanique** : Matrice géométrique d'interception et de blocage tactique (Theory of Games).
   - **Comment ça marche** : Son but est de provoquer la mort des concurrents directs pour éliminer la concurrence. Il scanne l'arène à la recherche d'adversaires dont la tête est à une distance $\le 4$ cases. 
   - **Interception de trajectoire** : Pour chaque cible proche, il identifie les deux prochaines positions probables de sa tête en prolongeant son vecteur de déplacement actuel ($\text{Tête} + 1 \times \text{Direction}$ et $\text{Tête} + 2 \times \text{Direction}$). Il oriente prioritairement son mouvement pour aller occuper ces cases cibles. S'il réussit, l'adversaire s'écrase sur son corps à la frame suivante, provoquant un "Kill". S'il n'y a aucun serpent à proximité, il se rabat sur une recherche de nourriture classique.
8. **Serpent 8 : L'Historien (Bleu indigo)** 
   - **Mécanique** : Apprentissage par instance via **K-Plus Proches Voisins (KNN)** avec vote majoritaire ($k=5$).
   - **Comment ça marche** : Il dispose d'une base de données de 10 000 états de jeu joués et enregistrés par un humain (`data/parties_humaines.pkl`). À chaque tick, il convertit sa situation actuelle en un vecteur binaire 11D (dangers relatifs, direction absolue, nourriture relative).
   - **Recherche par similarité** : Il effectue une distance euclidienne (équivalente à une distance de Hamming sur ce vecteur binaire) par rapport aux 10 000 exemples. Il extrait les 5 situations les plus proches, regarde quelles actions (tout droit, gauche, droite) l'humain avait prises, et prend la décision majoritaire.
   - **Filtre de sécurité** : Les données humaines ayant été enregistrées en mode solo sans collision possible avec d'autres serpents, si l'action votée par le KNN mène à une collision immédiate avec un adversaire dans l'arène MARL, l'agent outrepasse la décision humaine et applique une action de secours non-suicidaire.

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
