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
        
        # Compteurs
        self.nb_episodes = 0
        self.nb_ticks = 0
        
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


# =============================================================================
# 3. MINIMAX AGENT (SERPENT 5 - LE PSYCHOLOGUE)
# =============================================================================
class MinimaxAgent:
    """
    Agent Minimax avec élagage Alpha-Bêta de profondeur 3.
    Anticipe les mouvements du serpent adverse le plus proche pour bloquer ses trajectoires.
    """
    def __init__(self, profondeur=3):
        self.profondeur = profondeur
        
    def choisir_action(self, game, snake_id):
        snake = game.snakes[snake_id]
        head = snake['corps'][0]
        
        # Trouver l'opposant le plus proche
        closest_opp_id = None
        min_dist = float('inf')
        for opp_id, opp_s in game.snakes.items():
            if opp_id == snake_id:
                continue
            opp_head = opp_s['corps'][0]
            d = abs(opp_head[0] - head[0]) + abs(opp_head[1] - head[1])
            if d < min_dist:
                min_dist = d
                closest_opp_id = opp_id
                
        if closest_opp_id is None:
            # Fallback simple
            return AStarAgent().survie_fallback(game, snake_id)
            
        # Identifier les obstacles fixes (les corps des autres serpents)
        obstacles_fixes = set()
        for sid, s in game.snakes.items():
            if sid != snake_id and sid != closest_opp_id:
                obstacles_fixes.update(s['corps'])
                
        # Lancer la recherche Minimax (Max = Serpent 5, Min = Serpent le plus proche)
        best_score = -float('inf')
        best_action = 0
        alpha = -float('inf')
        beta = float('inf')
        
        # Les actions possibles pour Max
        for action in [0, 1, 2]:
            score = self.valeur_min(game, snake_id, closest_opp_id, action, self.profondeur - 1, alpha, beta, obstacles_fixes)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)
            
        return best_action
        
    def evaluer_etat(self, game, snake_id, opp_id, head_self, dir_self, body_self, head_opp, dir_opp, body_opp, obstacles_fixes):
        score = 0
        
        # 1. Vérification des collisions pour soi-même
        collision_self = False
        if not (0 <= head_self[0] < game.largeur and 0 <= head_self[1] < game.hauteur):
            collision_self = True
        elif head_self in obstacles_fixes or head_self in body_self[1:] or head_self in body_opp:
            collision_self = True
            
        # 2. Vérification des collisions pour l'ennemi
        collision_opp = False
        if not (0 <= head_opp[0] < game.largeur and 0 <= head_opp[1] < game.hauteur):
            collision_opp = True
        elif head_opp in obstacles_fixes or head_opp in body_opp[1:] or head_opp in body_self:
            collision_opp = True
            
        if collision_self:
            return -10000
        if collision_opp:
            score += 5000
            
        # 3. Distance à la pomme la plus proche
        if game.apples:
            closest_apple = min(game.apples, key=lambda a: abs(a[0] - head_self[0]) + abs(a[1] - head_self[1]))
            dist_apple = abs(closest_apple[0] - head_self[0]) + abs(closest_apple[1] - head_self[1])
            score += 100 / (dist_apple + 0.5)
            
        # 4. Stratégie d'anticipation et d'interception (bloquer l'opposant)
        dist_opp = abs(head_self[0] - head_opp[0]) + abs(head_self[1] - head_opp[1])
        if dist_opp <= 3:
            # Prédire le coup devant l'adversaire
            next_opp_pos = (head_opp[0] + dir_opp[0], head_opp[1] + dir_opp[1])
            dist_to_block = abs(head_self[0] - next_opp_pos[0]) + abs(head_self[1] - next_opp_pos[1])
            if dist_to_block == 1:
                score += 300  # Bonus de positionnement de blocage
                
        return score
        
    def valeur_max(self, game, snake_id, opp_id, depth, alpha, beta, obstacles_fixes, head_self, dir_self, body_self, head_opp, dir_opp, body_opp):
        if depth == 0:
            return self.evaluer_etat(game, snake_id, opp_id, head_self, dir_self, body_self, head_opp, dir_opp, body_opp, obstacles_fixes)
            
        v = -float('inf')
        idx_dir = get_dir_index(dir_self)
        
        for action in [0, 1, 2]:
            # Simulation mouvement de soi-même
            if action == 1:
                new_idx = (idx_dir - 1) % 4
            elif action == 2:
                new_idx = (idx_dir + 1) % 4
            else:
                new_idx = idx_dir
                
            new_dir = SENS_HORAIRE[new_idx]
            new_head = (head_self[0] + new_dir[0], head_self[1] + new_dir[1])
            new_body = [new_head] + body_self[:-1]
            
            score = self.valeur_min_sim(game, snake_id, opp_id, depth - 1, alpha, beta, obstacles_fixes, new_head, new_dir, new_body, head_opp, dir_opp, body_opp)
            v = max(v, score)
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
        
    def valeur_min_sim(self, game, snake_id, opp_id, depth, alpha, beta, obstacles_fixes, head_self, dir_self, body_self, head_opp, dir_opp, body_opp):
        if depth == 0:
            return self.evaluer_etat(game, snake_id, opp_id, head_self, dir_self, body_self, head_opp, dir_opp, body_opp, obstacles_fixes)
            
        v = float('inf')
        idx_dir_opp = get_dir_index(dir_opp)
        
        for action in [0, 1, 2]:
            # Simulation mouvement ennemi
            if action == 1:
                new_idx = (idx_dir_opp - 1) % 4
            elif action == 2:
                new_idx = (idx_dir_opp + 1) % 4
            else:
                new_idx = idx_dir_opp
                
            new_dir_opp = SENS_HORAIRE[new_idx]
            new_head_opp = (head_opp[0] + new_dir_opp[0], head_opp[1] + new_dir_opp[1])
            new_body_opp = [new_head_opp] + body_opp[:-1]
            
            score = self.valeur_max(game, snake_id, opp_id, depth - 1, alpha, beta, obstacles_fixes, head_self, dir_self, body_self, new_head_opp, new_dir_opp, new_body_opp)
            v = min(v, score)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
        
    def valeur_min(self, game, snake_id, opp_id, action_self, depth, alpha, beta, obstacles_fixes):
        # Initialisation racine du tour de Min (l'adversaire réagit à l'action de Max)
        snake = game.snakes[snake_id]
        opp_s = game.snakes[opp_id]
        
        head_self = snake['corps'][0]
        dir_self = snake['direction']
        idx_dir = get_dir_index(dir_self)
        
        if action_self == 1:
            new_idx = (idx_dir - 1) % 4
        elif action_self == 2:
            new_idx = (idx_dir + 1) % 4
        else:
            new_idx = idx_dir
            
        new_dir_self = SENS_HORAIRE[new_idx]
        new_head_self = (head_self[0] + new_dir_self[0], head_self[1] + new_dir_self[1])
        new_body_self = [new_head_self] + snake['corps'][:-1]
        
        v = float('inf')
        head_opp = opp_s['corps'][0]
        dir_opp = opp_s['direction']
        idx_dir_opp = get_dir_index(dir_opp)
        
        for action in [0, 1, 2]:
            if action == 1:
                new_idx_opp = (idx_dir_opp - 1) % 4
            elif action == 2:
                new_idx_opp = (idx_dir_opp + 1) % 4
            else:
                new_idx_opp = idx_dir_opp
                
            new_dir_opp = SENS_HORAIRE[new_idx_opp]
            new_head_opp = (head_opp[0] + new_dir_opp[0], head_opp[1] + new_dir_opp[1])
            new_body_opp = [new_head_opp] + opp_s['corps'][:-1]
            
            score = self.valeur_max(game, snake_id, opp_id, depth, alpha, beta, obstacles_fixes,
                                   new_head_self, new_dir_self, new_body_self,
                                   new_head_opp, new_dir_opp, new_body_opp)
            v = min(v, score)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v


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


# =============================================================================
# 6. KNN HISTORIAN AGENT (SERPENT 8 - L'HISTORIEN)
# =============================================================================
class HistorianAgent:
    """
    Agent sans IA utilisant KNN (K-Nearest Neighbors).
    Recherche les situations de l'état 11D les plus similaires dans 'parties_humaines.pkl'
    et reproduit l'action de l'humain.
    """
    def __init__(self, chemin_donnees="data/parties_humaines.pkl", k=5):
        self.k = k
        self.etats_historique = None
        self.actions_historique = None
        
        if os.path.exists(chemin_donnees):
            try:
                with open(chemin_donnees, "rb") as f:
                    donnees = pickle.load(f)
                    
                # Validation des données
                donnees_valides = [
                    d for d in donnees 
                    if isinstance(d, tuple) and len(d) == 2 
                    and isinstance(d[0], np.ndarray) and len(d[0]) == 11
                ]
                
                if donnees_valides:
                    # Limiter la taille pour garder une recherche ultra rapide (max 10 000 transitions)
                    if len(donnees_valides) > 10000:
                        donnees_valides = random.sample(donnees_valides, 10000)
                        
                    self.etats_historique = np.array([x[0] for x in donnees_valides], dtype=np.float32)
                    self.actions_historique = np.array([x[1] for x in donnees_valides], dtype=np.int64)
                    print(f"[✓] Historien : {len(self.etats_historique)} transitions chargées pour le KNN.")
                else:
                    print("[!] Historien : Fichier pkl trouvé mais sans données d'état 11D valides.")
            except Exception as e:
                print(f"[!] Historien : Erreur lors de la lecture du fichier KNN : {e}")
        else:
            print(f"[!] Historien : Fichier {chemin_donnees} absent. Mode dégradé (Économiste).")

    def choisir_action(self, game, snake_id):
        # Fallback si pas de données historiques
        if self.etats_historique is None or len(self.etats_historique) == 0:
            return EconomistAgent().choisir_action(game, snake_id)
            
        # Calculer le vecteur d'état 11D de l'agent historien
        etat_courant = game.get_11d_state(snake_id)
        
        # Calcul vectorisé rapide des distances (L2 au carré, équivalent à Hamming sur des variables binaires)
        distances = np.sum((self.etats_historique - etat_courant) ** 2, axis=1)
        
        if self.k == 1:
            meilleur_idx = np.argmin(distances)
            action = self.actions_historique[meilleur_idx]
        else:
            # Sélectionner les k indices les plus proches
            indices_proches = np.argpartition(distances, self.k)[:self.k]
            actions_proches = self.actions_historique[indices_proches]
            
            # Vote à la majorité
            votes = np.bincount(actions_proches, minlength=3)
            action = np.argmax(votes)
            
        # Filtre de sécurité : si l'action copiée de l'humain mène à une mort immédiate
        # (les humains jouaient seuls, sans concurrents), on cherche un repli sécurisé.
        next_pos = game.get_next_cell_position(snake_id, action)
        if game.est_collision(next_pos, snake_id):
            for action_secours in [0, 1, 2]:
                if action_secours != action:
                    secours_pos = game.get_next_cell_position(snake_id, action_secours)
                    if not game.est_collision(secours_pos, snake_id):
                        return action_secours
                        
        return action
