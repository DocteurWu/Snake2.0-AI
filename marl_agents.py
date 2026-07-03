# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA MARL — Les 8 Architectures d'Agents
=============================================================================
 Implémentation des 8 agents (3 DQN PyTorch, 5 algorithmiques/mathématiques).
=============================================================================
"""

import os
import random
import pickle
import heapq
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from marl_model import ReseauDQN

# Constantes de directions et de jeu
DROITE = (1, 0)
BAS = (0, 1)
GAUCHE = (-1, 0)
HAUT = (0, -1)
SENS_HORAIRE = [DROITE, BAS, GAUCHE, HAUT]

def get_dir_index(direction):
    """Retourne l'index d'une direction dans SENS_HORAIRE."""
    try:
        return SENS_HORAIRE.index(direction)
    except ValueError:
        return 0

# =============================================================================
# 1. DQN AGENT (SERPENTS 1, 2, 3)
# =============================================================================
class DQNAgent:
    """
    Agent DQN générique réutilisable pour les Serpents 1, 2 et 3.
    Gère son propre réseau principal, réseau cible, replay buffer et apprentissage.
    """
    def __init__(self, nom, taille_entree, couches_cachees, lr=0.001, epsilon_decay=0.999):
        self.nom = nom
        self.device = torch.device("cpu")
        
        # Réseaux
        self.reseau_principal = ReseauDQN(taille_entree, couches_cachees).to(self.device)
        self.reseau_cible = ReseauDQN(taille_entree, couches_cachees).to(self.device)
        self.reseau_cible.load_state_dict(self.reseau_principal.state_dict())
        self.reseau_cible.eval()
        
        # Optimisation
        self.optimiseur = optim.Adam(self.reseau_principal.parameters(), lr=lr)
        self.critere = nn.MSELoss()
        
        # Mémoire Replay
        self.memoire = deque(maxlen=20000)
        self.taille_batch = 64
        self.gamma = 0.99
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_fin = 0.05
        self.epsilon_decroissance = epsilon_decay
        
        # Sauvegarde
        self.chemin_sauvegarde = f"models/marl_dqn_{self.nom.lower()}.pth"
        self.meilleure_moyenne = 0.0
        
        # Compteurs
        self.nb_episodes = 0
        self.nb_ticks = 0
        self.total_transitions = 0
        
    def choisir_action(self, etat, entrainement=True):
        """Choisit une action relative (0=tout droit, 1=gauche, 2=droite) via epsilon-greedy."""
        if entrainement and random.random() < self.epsilon:
            return random.randint(0, 2)
            
        with torch.no_grad():
            etat_tensor = torch.tensor(etat, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.reseau_principal(etat_tensor)
            return q_values.argmax(dim=1).item()
            
    def memoriser(self, etat, action, recompense, etat_suivant, termine):
        """Ajoute une transition dans le replay buffer."""
        self.memoire.append((etat, action, recompense, etat_suivant, termine))
        self.total_transitions += 1
        
    def entrainer(self):
        """Effectue une étape d'apprentissage DQN sur un batch aléatoire."""
        if len(self.memoire) < self.taille_batch:
            return
            
        # Échantillonnage
        batch = random.sample(self.memoire, self.taille_batch)
        etats, actions, recompenses, etats_suivants, termines = zip(*batch)
        
        etats = torch.tensor(np.array(etats), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        recompenses = torch.tensor(recompenses, dtype=torch.float32, device=self.device)
        etats_suivants = torch.tensor(np.array(etats_suivants), dtype=torch.float32, device=self.device)
        termines = torch.tensor(termines, dtype=torch.float32, device=self.device)
        
        # Q-values actuelles estimées par le réseau principal
        q_values = self.reseau_principal(etats).gather(1, actions).squeeze(1)
        
        # Q-values cibles estimées par le réseau cible (Double DQN simplifié)
        with torch.no_grad():
            next_q_values = self.reseau_cible(etats_suivants).max(dim=1)[0]
            targets = recompenses + (1 - termines) * self.gamma * next_q_values
            
        loss = self.critere(q_values, targets)
        
        self.optimiseur.zero_grad()
        loss.backward()
        # Gradient clipping pour stabiliser
        nn.utils.clip_grad_norm_(self.reseau_principal.parameters(), max_norm=1.0)
        self.optimiseur.step()
        
        self.nb_ticks += 1
        
    def fin_episode(self):
        """Appelé à chaque fois que l'agent meurt pour mettre à jour epsilon et synchroniser le réseau cible."""
        self.nb_episodes += 1
        # Décroissance d'epsilon
        self.epsilon = max(self.epsilon_fin, self.epsilon * self.epsilon_decroissance)
        
        # Synchronisation périodique du réseau cible
        if self.nb_episodes % 10 == 0:
            self.reseau_cible.load_state_dict(self.reseau_principal.state_dict())

    def charger_si_existe(self):
        """Charge les poids s'il y a une sauvegarde, et adapte epsilon et statistiques en conséquence."""
        charge = False
        if os.path.exists(self.chemin_sauvegarde):
            charge = self.reseau_principal.charger(self.chemin_sauvegarde, self.device)
        elif self.nom == "Standard" and os.path.exists("models/dqn_snake.pth"):
            # Fallback vers le modèle pré-entraîné solo pour le DQN standard
            charge = self.reseau_principal.charger("models/dqn_snake.pth", self.device)
            
        if charge:
            self.reseau_cible.load_state_dict(self.reseau_principal.state_dict())
            self.epsilon = 0.25
            
            # Charger aussi les métadonnées pour continuer la décroissance d'epsilon et la moyenne
            chemin_meta = self.chemin_sauvegarde.replace(".pth", "_meta.pkl")
            if os.path.exists(chemin_meta):
                try:
                    with open(chemin_meta, "rb") as f:
                        meta = pickle.load(f)
                    self.meilleure_moyenne = meta.get('meilleure_moyenne', 0.0)
                    self.nb_episodes = meta.get('nb_episodes', 0)
                    self.epsilon = meta.get('epsilon', 0.25)
                    self.total_transitions = meta.get('total_transitions', 0)
                    print(f"[Arena] [{self.nom}] Métadonnées chargées : Meilleure moyenne = {self.meilleure_moyenne:.2f}, Episodes = {self.nb_episodes}, Epsilon = {self.epsilon:.3f}, Données collectées = {self.total_transitions}")
                except Exception as e:
                    print(f"[Arena] [{self.nom}] Erreur lecture métadonnées : {e}")
            else:
                print(f"[Arena] [{self.nom}] Pas de métadonnées trouvées. Utilisation d'un epsilon par défaut de 0.25.")
            return True
        return False

    def sauvegarder_si_meilleur(self, moyenne_actuelle):
        """Sauvegarde les poids du modèle si sa moyenne actuelle de pommes par vie bat son record."""
        # On attend au moins 5 vies pour que la moyenne soit significative
        if self.nb_episodes >= 5 and moyenne_actuelle > self.meilleure_moyenne:
            self.meilleure_moyenne = moyenne_actuelle
            try:
                self.reseau_principal.sauvegarder(self.chemin_sauvegarde)
                
                # Sauvegarder les métadonnées pour la reprise
                chemin_meta = self.chemin_sauvegarde.replace(".pth", "_meta.pkl")
                with open(chemin_meta, "wb") as f:
                    pickle.dump({
                        'meilleure_moyenne': self.meilleure_moyenne,
                        'nb_episodes': self.nb_episodes,
                        'epsilon': self.epsilon,
                        'total_transitions': self.total_transitions
                    }, f)
                print(f"[Record] >>> Agent [{self.nom}] bat son record avec {moyenne_actuelle:.2f} pommes/vie ! Poids sauvegardés.")
            except Exception as e:
                print(f"[!] Erreur de sauvegarde pour [{self.nom}] : {e}")


# =============================================================================
# 2. A* PATHFINDER (SERPENT 4 - LE MATHEMATICIEN)
# =============================================================================
class AStarAgent:
    """
    Agent calculant le chemin le plus court vers la pomme la plus proche en utilisant A*.
    Sans apprentissage.
    """
    def choisir_action(self, game, snake_id):
        snake = game.snakes[snake_id]
        head = snake['corps'][0]
        
        # Trouver la pomme la plus proche
        if not game.apples:
            return 0
        closest_apple = min(game.apples, key=lambda a: abs(a[0] - head[0]) + abs(a[1] - head[1]))
        
        # Calculer le chemin A*
        path = self.calculer_astar(game, head, closest_apple, snake_id)
        
        if path and len(path) > 1:
            next_cell = path[1]
            return self.traduire_cellule_en_action(snake, next_cell)
        else:
            # Fallback de survie si aucun chemin n'est trouvé
            return self.survie_fallback(game, snake_id)
            
    def calculer_astar(self, game, start, target, snake_id):
        # File de priorité : (f_score, position)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: abs(target[0] - start[0]) + abs(target[1] - start[1])}
        
        in_open_set = {start}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            in_open_set.remove(current)
            
            if current == target:
                # Reconstruire le chemin
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
                
            for neighbor in self.obtenir_voisins(game, current, snake_id):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + abs(target[0] - neighbor[0]) + abs(target[1] - neighbor[1])
                    
                    if neighbor not in in_open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        in_open_set.add(neighbor)
                        
        return None
        
    def obtenir_voisins(self, game, cell, snake_id):
        voisins = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cell[0] + dx, cell[1] + dy
            if 0 <= nx < game.largeur and 0 <= ny < game.hauteur:
                # Éviter les collisions avec les corps des serpents
                obstacle = False
                for sid, s in game.snakes.items():
                    if (nx, ny) in s['corps']:
                        obstacle = True
                        break
                if not obstacle:
                    voisins.append((nx, ny))
        return voisins
        
    def traduire_cellule_en_action(self, snake, target_cell):
        head = snake['corps'][0]
        target_dir = (target_cell[0] - head[0], target_cell[1] - head[1])
        
        idx_courant = get_dir_index(snake['direction'])
        idx_cible = get_dir_index(target_dir)
        
        if idx_cible == idx_courant:
            return 0  # Tout droit
        elif idx_cible == (idx_courant - 1) % 4:
            return 1  # Tourner à gauche
        elif idx_cible == (idx_courant + 1) % 4:
            return 2  # Tourner à droite
            
        return 0  # Par défaut
        
    def survie_fallback(self, game, snake_id):
        """Choisit l'action qui offre le plus de voisins libres à l'étape suivante."""
        best_action = 0
        max_voisins_libres = -1
        
        for action in [0, 1, 2]:
            next_pos = game.get_next_cell_position(snake_id, action)
            if not game.est_collision(next_pos, snake_id):
                voisins = self.obtenir_voisins(game, next_pos, snake_id)
                if len(voisins) > max_voisins_libres:
                    max_voisins_libres = len(voisins)
                    best_action = action
        return best_action


# MinimaxAgent supprimé au profit du modèle Élite (DQN Sélectif).


# =============================================================================
# 4. POTENTIAL MAP AGENT (SERPENT 6 - L'ECONOMISTE)
# =============================================================================
class EconomistAgent:
    """
    Agent basé sur une carte thermique d'influence.
    Les pommes attirent (+1), les obstacles/corps/murs repoussent (-5).
    """
    def choisir_action(self, game, snake_id):
        best_action = 0
        best_potential = -float('inf')
        
        for action in [0, 1, 2]:
            next_pos = game.get_next_cell_position(snake_id, action)
            
            if game.est_collision(next_pos, snake_id):
                potential = -1e9
            else:
                potential = 0.0
                
                # Attraction des pommes
                for apple in game.apples:
                    d = abs(apple[0] - next_pos[0]) + abs(apple[1] - next_pos[1])
                    potential += 1.0 / (d + 0.5)
                    
                # Répulsion des autres serpents (tous les segments)
                for sid, s in game.snakes.items():
                    for idx, segment in enumerate(s['corps']):
                        d = abs(segment[0] - next_pos[0]) + abs(segment[1] - next_pos[1])
                        # Pénalité plus forte pour les têtes adverses à proximité
                        coeff = 10.0 if (idx == 0 and sid != snake_id) else 5.0
                        potential -= coeff / (d + 0.5)
                        
                # Répulsion des murs
                dist_mur_x = min(next_pos[0], game.largeur - 1 - next_pos[0])
                dist_mur_y = min(next_pos[1], game.hauteur - 1 - next_pos[1])
                dist_mur = min(dist_mur_x, dist_mur_y)
                potential -= 5.0 / (dist_mur + 0.5)
                
            if potential > best_potential:
                best_potential = potential
                best_action = action
                
        return best_action


# =============================================================================
# 5. STRATEGIST AGENT (SERPENT 7 - LE STRATEGE)
# =============================================================================
class StrategistAgent:
    """
    Agent basé sur la théorie des jeux pure.
    Cherche prioritairement à couper la route des autres serpents pour provoquer des crashs ("Kills").
    """
    def choisir_action(self, game, snake_id):
        snake = game.snakes[snake_id]
        head = snake['corps'][0]
        
        # Trouver les opposants proches (distance de Manhattan <= 4)
        opposants_proches = []
        for opp_id, opp_s in game.snakes.items():
            if opp_id == snake_id:
                continue
            opp_head = opp_s['corps'][0]
            d = abs(opp_head[0] - head[0]) + abs(opp_head[1] - head[1])
            if d <= 4:
                opposants_proches.append((opp_s, d))
                
        best_action = 0
        best_score = -float('inf')
        
        for action in [0, 1, 2]:
            next_pos = game.get_next_cell_position(snake_id, action)
            
            if game.est_collision(next_pos, snake_id):
                score = -1e9
            else:
                # Score de base : s'orienter vers la pomme la plus proche
                score = 0.0
                if game.apples:
                    closest_apple = min(game.apples, key=lambda a: abs(a[0] - next_pos[0]) + abs(a[1] - next_pos[1]))
                    dist_apple = abs(closest_apple[0] - next_pos[0]) + abs(closest_apple[1] - next_pos[1])
                    score -= dist_apple * 0.5
                    
                # Analyse géométrique pour provoquer un crash
                for opp_s, d in opposants_proches:
                    opp_head = opp_s['corps'][0]
                    opp_dir = opp_s['direction']
                    
                    # Cellules de trajectoire probable de l'ennemi
                    projete_1 = (opp_head[0] + opp_dir[0], opp_head[1] + opp_dir[1])
                    projete_2 = (opp_head[0] + 2 * opp_dir[0], opp_head[1] + 2 * opp_dir[1])
                    
                    # Si notre action nous place sur la route projetée, on l'intercepte !
                    if next_pos == projete_1:
                        score += 150.0  # Blocage imminent !
                    elif next_pos == projete_2:
                        score += 80.0
                        
                    # Éviter de s'écraser bêtement si on est juste à côté de leur corps
                    for segment in opp_s['corps'][1:]:
                        if abs(segment[0] - next_pos[0]) + abs(segment[1] - next_pos[1]) <= 1:
                            score -= 20.0
                            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action


# HistorianAgent supprimé au profit du modèle Collectif (DQN Partagé).
