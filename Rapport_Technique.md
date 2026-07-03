# Modélisation Mathématique & Logique Technique du Projet Snake IA MARL

Ce document décrit en détail les fondements mathématiques, les architectures d'agents et la logique technique qui régissent l'arène de simulation Multi-Agent Reinforcement Learning (MARL) compétitive synchrone développée dans ce projet.

---

## 1. Processus de Décision Markovien (MDP)

Le jeu multi-agent synchrone est modélisé comme un **Processus de Décision Markovien** étendu au cadre multi-agent, défini par le tuple $(N, S, \{A_i\}_{i=1}^N, P, \{R_i\}_{i=1}^N, \gamma)$ où $N = 8$ est le nombre d'agents simultanés.

### Espace d'États $S$
Les agents DQN utilisent deux représentations distinctes de l'espace des états :

1.  **Vecteur d'État 11D (DQN Standard, DQN Profond, DQN Élite, DQN Collectif)** :
    Le vecteur d'état $s_t \in \mathbb{R}^{11}$ d'un serpent $i$ à l'instant $t$ comprend :
    *   **Dangers immédiats (3D)** : $s_t[0..2] \in \{0, 1\}^3$ (Devant, Gauche, Droite relative par rapport à la direction de la tête).
    *   **Direction absolue de marche (4D)** : $s_t[3..6] \in \{0, 1\}^4$ (Vecteur One-Hot parmi Nord, Sud, Est, Ouest).
    *   **Direction relative de la pomme la plus proche (4D)** : $s_t[7..10] \in \{0, 1\}^4$ (Vecteur One-Hot indiquant si la pomme la plus proche est en Devant, Derrière, Gauche ou Droite relative par rapport à la tête et son orientation).

2.  **Vecteur Raycast 24D (DQN Raycast)** :
    Le vecteur d'état $s_t \in \mathbb{R}^{24}$ est composé de 8 rayons projetés radialement depuis la tête (Nord, Nord-Est, Est, Sud-Est, Sud, Sud-Ouest, Ouest, Nord-Ouest, décalés selon l'orientation courante de la tête). Chaque rayon renvoie 3 caractéristiques normalisées entre $0$ et $1$ :
    $$Ray_d = \left[ \frac{1}{\text{dist\_mur}}, \frac{1}{\text{dist\_pomme}}, \frac{1}{\text{dist\_corps}} \right]$$

### Espace d'Actions $A_i$ (Dimension 3)
Chaque agent choisit à chaque tick une action dans son espace d'actions relatives :
$$A_i = \{0 \text{ (Tout droit)}, 1 \text{ (Tourner à Gauche)}, 2 \text{ (Tourner à Droite)}\}$$

### Fonction de Récompense $R_i(s_t, a_t)$
Chaque agent est récompensé de manière indépendante avec du *reward shaping* pour guider son exploration :
$$R_i(s_t, a_t) = \begin{cases} 
  +10.0 & \text{si l'agent $i$ mange une pomme} \\
  -10.0 & \text{si l'agent $i$ meurt (collision ou inanition)} \\
  +1.0  & \text{si la distance de Manhattan } d_M(\text{tête}_i, \text{pomme\_proche}) \text{ diminue} \\
  -1.0  & \text{si la distance de Manhattan } d_M(\text{tête}_i, \text{pomme\_proche}) \text{ augmente} \\
  0.0   & \text{sinon}
\end{cases}$$

---

## 2. Deep Q-Learning (DQN)

Les agents DQN approximment la fonction de valeur optimale Action-Valeur $Q^*(s, a)$ par un réseau de neurones paramétré par $\theta$ :
$$Q(s, a; \theta) \approx Q^*(s, a)$$

### Perte et Optimisation (Loss MSE)
La perte quadratique moyenne minimisée à chaque étape d'apprentissage à partir d'un batch de taille $B=64$ extrait du Replay Buffer de l'agent est :
$$\mathcal{L}_{DQN}(\theta) = \frac{1}{B} \sum_{k=1}^B \left( y_k - Q(s_k, a_k; \theta) \right)^2$$

Où la cible $y_k$ (TD-target) est calculée via le réseau cible (Target Network) $\theta^-$ :
$$y_k = r_k + \gamma (1 - d_k) \max_{a'} Q(s'_k, a'; \theta^-)$$
($d_k = 1$ si l'état suivant $s'_k$ est terminal, 0 sinon). Le réseau cible est mis à jour tous les 10 épisodes : $\theta^- \leftarrow \theta$.

---

## 3. Apprentissage Centralisé : Le DQN Collectif

Le **DQN Collectif (Serpent 8)** implémente un paradigme d'apprentissage centralisé à partir de données partagées.

### Logique du Replay Buffer Partagé
Contrairement aux DQN standards qui n'apprennent qu'à partir de leurs propres transitions, le DQN Collectif collecte les transitions de **tous les 8 agents** (algorithmiques et IA) présents dans la simulation à chaque tick $t$ :
$$\mathcal{D}_{\text{Collectif}} \leftarrow \mathcal{D}_{\text{Collectif}} \cup \left\{ (s_t^j, a_t^j, r_t^j, s_{t+1}^j, d_t^j) \right\}_{j=0}^7$$

Le modèle s'entraîne donc sur une distribution de données extrêmement riche, combinant les trajectoires parfaites des heuristiques mathématiques (A*, Influence) et les phases exploratoires des réseaux DQN, ce qui accélère sa convergence face à la diversité de l'arène.

---

## 4. Apprentissage par Imitation Sélective : Le DQN Élite

L'agent **Élite (Serpent 5)** implémente un apprentissage sélectif par renforcement et imitation de l'expert en temps réel.

### Sélection de l'Expert
À chaque tick $t$, le jeu identifie l'agent ayant obtenu le meilleur score de performance actuel (moyenne de pommes sur les 10 dernières vies) :
$$i^* = \arg\max_{j \in \{0..7\}} M_{10}(j)$$

### Alimentation du Buffer Élite
L'agent Élite (agent 4) stocke dans son Replay Buffer $\mathcal{D}_{\text{Élite}}$ :
1.  Sa propre transition $(s_t^4, a_t^4, r_t^4, s_{t+1}^4, d_t^4)$.
2.  La transition de l'expert $i^*$ $(s_t^{i^*}, a_t^{i^*}, r_t^{i^*}, s_{t+1}^{i^*}, d_t^{i^*})$ si $i^* \neq 4$.

Ce mécanisme permet d'exploiter dynamiquement les stratégies gagnantes présentes dans l'arène sans souffrir des mauvaises données générées par les agents en cours d'exploration ou en échec.

---

## 5. Modélisation Mathématique des Agents Algorithmiques

### A) Mathématicien (A* Search)
L'agent calcule à chaque tick un chemin optimal sur un graphe discret délimité par la grille de $40 \times 40$ cases, en utilisant la distance de Manhattan comme heuristique admissible $h(n)$ :
$$f(n) = g(n) + h(n)$$
$$h(n) = |x_n - x_{\text{pomme}}| + |y_n - y_{\text{pomme}}|$$
Où $g(n)$ est le coût accumulé depuis le nœud de départ (tête). Si aucun chemin n'existe, un algorithme de repli choisit l'action maximisant l'espace de liberté (nombre de voisins libres au pas suivant).

### B) Économiste (Influence Map / Heatmap)
L'agent évalue chaque case adjacente $c \in \{\text{Devant}, \text{Gauche}, \text{Droite}\}$ en calculant son potentiel $V(c)$ :
$$V(c) = \sum_{a \in \text{Pommes}} \frac{1}{d_M(c, a) + 0.5} - \sum_{o \in \text{Obstacles}} \frac{5}{d_M(c, o)^2 + 0.1} - \sum_{h \in \text{Têtes Opposants}} \frac{10}{d_M(c, h)^2 + 0.1}$$
L'action choisie est celle qui maximise $V(c)$.

### C) Stratège (Game Theory / Lookahead)
L'agent simule les trajectoires futures possibles des têtes adverses situées dans un rayon de 4 cases. Il projette les positions à $t+1$ et $t+2$ pour planifier un blocage direct (couper la trajectoire) afin de provoquer des collisions adverses tout en garantissant sa propre sécurité.

---

## 6. Métrique d'Évaluation : Moyenne Glissante sur 10 Vies

Afin de refléter fidèlement l'état de performance actuel d'un agent sans pénaliser les phases initiales d'exploration aléatoire, le classement et les décisions de sauvegarde des réseaux DQN reposent sur la **moyenne des scores des 10 dernières vies** :

$$M_{10}(i) = \frac{1}{|H_i|} \sum_{s \in H_i} s$$

Où $H_i$ est la file circulaire (`deque` de taille maximale 10) contenant les scores (nombre de pommes mangées) obtenus par le serpent $i$ lors de ses 10 dernières morts. Si un agent n'est pas encore mort, le score de sa vie courante est utilisé comme valeur par défaut.
