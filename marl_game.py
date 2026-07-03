# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA MARL — Moteur de Jeu Multi-Agent
=============================================================================
 Gère la grille de 40x40, les 8 pommes actives, le déplacement synchrone,
 la détection des collisions complexes et le respawn continu.
=============================================================================
"""

import random
import numpy as np

# Constantes de directions
DROITE = (1, 0)
BAS = (0, 1)
GAUCHE = (-1, 0)
HAUT = (0, -1)
SENS_HORAIRE = [DROITE, BAS, GAUCHE, HAUT]

class MARLGame:
    """
    Moteur de jeu Snake pour simulation Multi-Agent Reinforcement Learning (MARL).
    """
    def __init__(self, largeur=40, hauteur=40, nb_pommes=8):
        self.largeur = largeur
        self.hauteur = hauteur
        self.nb_pommes = nb_pommes
        
        # Données de l'arène
        self.snakes = {}
        self.apples = []
        
        self.reset()
        
    def reset(self):
        """Réinitialise l'arène (vide de pommes et de serpents)."""
        self.apples = []
        self.snakes = {}
        
    def ajouter_serpent(self, snake_id, nom, couleur):
        """Initialise et ajoute un serpent dans l'arène."""
        self.snakes[snake_id] = {
            'id': snake_id,
            'nom': nom,
            'couleur': couleur,
            'corps': [],  # Sera rempli lors du respawn
            'direction': (1, 0),
            'index_direction': 0,
            'score_courant': 0,
            'nb_respawns': 0,
            'total_pommes_historique': 0,
            'max_longueur': 3,
            'steps_sans_manger': 0
        }
        self.respawn_serpent(snake_id)
        
    def initialiser_pommes(self):
        """Génère le lot initial de pommes dans l'arène."""
        self.apples = []
        for _ in range(self.nb_pommes):
            self.placer_nouvelle_pomme()
            
    def placer_nouvelle_pomme(self):
        """Place une pomme sur une cellule vide aléatoire."""
        cases_vide = self.obtenir_cases_vides()
        if cases_vide:
            self.apples.append(random.choice(cases_vide))
            
    def obtenir_cases_vides(self):
        """Retourne la liste des cellules (x, y) libres (sans corps ni pomme)."""
        cases_occupees = set(self.apples)
        for s in self.snakes.values():
            cases_occupees.update(s['corps'])
            
        cases_vides = []
        for x in range(self.largeur):
            for y in range(self.hauteur):
                if (x, y) not in cases_occupees:
                    cases_vides.append((x, y))
        return cases_vides
        
    def respawn_serpent(self, snake_id):
        """
        Téléporte immédiatement un serpent sur une zone libre aléatoire.
        Cherche un espace linéaire de 3 cellules pour étendre le corps de manière sûre.
        """
        cases_vides = self.obtenir_cases_vides()
        if not cases_vides:
            # Fallback absolu si la grille est saturée (très improbable)
            x = random.randint(5, self.largeur - 6)
            y = random.randint(5, self.hauteur - 6)
            dir_random = random.choice(SENS_HORAIRE)
            idx_dir = SENS_HORAIRE.index(dir_random)
            self.snakes[snake_id]['corps'] = [(x, y), (x, y), (x, y)]
            self.snakes[snake_id]['direction'] = dir_random
            self.snakes[snake_id]['index_direction'] = idx_dir
            self.snakes[snake_id]['steps_sans_manger'] = 0
            return
            
        random.shuffle(cases_vides)
        for tete_pos in cases_vides:
            tx, ty = tete_pos
            # Mélanger les directions pour essayer de positionner les segments arrière
            dirs = SENS_HORAIRE[:]
            random.shuffle(dirs)
            for direction in dirs:
                dx, dy = direction
                seg_1 = (tx - dx, ty - dy)
                seg_2 = (tx - 2 * dx, ty - 2 * dy)
                
                # S'assurer que le corps est bien dans la grille
                if (0 <= seg_1[0] < self.largeur and 0 <= seg_1[1] < self.hauteur and
                    0 <= seg_2[0] < self.largeur and 0 <= seg_2[1] < self.hauteur):
                    
                    # Vérifier si ces cases sont occupées par d'autres serpents ou pommes
                    occupe = False
                    for sid, s in self.snakes.items():
                        if sid == snake_id:
                            continue
                        if tete_pos in s['corps'] or seg_1 in s['corps'] or seg_2 in s['corps']:
                            occupe = True
                            break
                    if tete_pos in self.apples or seg_1 in self.apples or seg_2 in self.apples:
                        occupe = True
                        
                    if not occupe:
                        # Zone de respawn parfaite trouvée !
                        self.snakes[snake_id]['corps'] = [tete_pos, seg_1, seg_2]
                        self.snakes[snake_id]['direction'] = direction
                        self.snakes[snake_id]['index_direction'] = SENS_HORAIRE.index(direction)
                        self.snakes[snake_id]['steps_sans_manger'] = 0
                        return
                        
        # Si aucun alignement parfait n'a fonctionné, repli sur une cellule isolée (corps replié)
        tx, ty = cases_vides[0]
        dir_random = random.choice(SENS_HORAIRE)
        self.snakes[snake_id]['corps'] = [(tx, ty), (tx, ty), (tx, ty)]
        self.snakes[snake_id]['direction'] = dir_random
        self.snakes[snake_id]['index_direction'] = SENS_HORAIRE.index(dir_random)
        self.snakes[snake_id]['steps_sans_manger'] = 0

    def est_collision(self, cell, snake_id):
        """Vérifie si une cellule (x, y) est en collision avec un mur ou un serpent."""
        x, y = cell
        if x < 0 or x >= self.largeur or y < 0 or y >= self.hauteur:
            return True
            
        for sid, s in self.snakes.items():
            # Collision avec le corps actuel
            if cell in s['corps']:
                return True
        return False
        
    def get_next_cell_position(self, snake_id, action):
        """Calcule la position de la tête du serpent s'il choisit une action relative."""
        snake = self.snakes[snake_id]
        head = snake['corps'][0]
        idx_dir = snake['index_direction']
        
        if action == 1:    # Gauche
            idx_dir = (idx_dir - 1) % 4
        elif action == 2:  # Droite
            idx_dir = (idx_dir + 1) % 4
            
        dx, dy = SENS_HORAIRE[idx_dir]
        return (head[0] + dx, head[1] + dy)
        
    def get_position_relative(self, snake_id, direction_relative):
        """Calcule la position adjacente à la tête selon une direction relative."""
        snake = self.snakes[snake_id]
        head = snake['corps'][0]
        idx_dir = snake['index_direction']
        
        if direction_relative == 'devant':
            idx = idx_dir
        elif direction_relative == 'gauche':
            idx = (idx_dir - 1) % 4
        elif direction_relative == 'droite':
            idx = (idx_dir + 1) % 4
        else:
            idx = idx_dir
            
        dx, dy = SENS_HORAIRE[idx]
        return (head[0] + dx, head[1] + dy)

    # =============================================================================
    # CALCULS DE VECTEURS D'ETAT
    # =============================================================================
    def get_11d_state(self, snake_id):
        """
        Calcule l'état 11D classique (identique au jeu solo d'origine).
        Relatif à la pomme la plus proche.
        """
        snake = self.snakes[snake_id]
        head = snake['corps'][0]
        direction = snake['direction']
        idx_dir = snake['index_direction']
        
        # Dangers relatifs (devant, gauche, droite)
        danger_devant = int(self.est_collision(self.get_position_relative(snake_id, 'devant'), snake_id))
        danger_gauche = int(self.est_collision(self.get_position_relative(snake_id, 'gauche'), snake_id))
        danger_droite = int(self.est_collision(self.get_position_relative(snake_id, 'droite'), snake_id))
        
        # Direction absolue actuelle (One-Hot)
        dir_droite = int(direction == DROITE)
        dir_bas    = int(direction == BAS)
        dir_gauche = int(direction == GAUCHE)
        dir_haut   = int(direction == HAUT)
        
        # Position nourriture relative (pomme la plus proche)
        if not self.apples:
            food_devant = food_derriere = food_gauche = food_droite = 0
        else:
            closest_apple = min(self.apples, key=lambda a: abs(a[0] - head[0]) + abs(a[1] - head[1]))
            dx_food = closest_apple[0] - head[0]
            dy_food = closest_apple[1] - head[1]
            
            dir_x, dir_y = direction
            dir_gauche_vec = SENS_HORAIRE[(idx_dir - 1) % 4]
            dir_droite_vec = SENS_HORAIRE[(idx_dir + 1) % 4]
            
            food_devant   = int((dx_food * dir_x + dy_food * dir_y) > 0)
            food_derriere = int((dx_food * dir_x + dy_food * dir_y) < 0)
            food_gauche   = int((dx_food * dir_gauche_vec[0] + dy_food * dir_gauche_vec[1]) > 0)
            food_droite   = int((dx_food * dir_droite_vec[0] + dy_food * dir_droite_vec[1]) > 0)
            
        return np.array([
            danger_devant, danger_gauche, danger_droite,
            dir_droite, dir_bas, dir_gauche, dir_haut,
            food_devant, food_derriere, food_gauche, food_droite
        ], dtype=np.float32)

    def get_raycast_state(self, snake_id):
        """
        Vecteur d'état 24D basé sur la vision "lasers" (8 directions).
        Pour chaque direction (relative horaire par rapport à la tête) :
        [distance_mur, distance_pomme, distance_corps].
        Valeurs normalisées en 1 / distance (0 si introuvable).
        """
        snake = self.snakes[snake_id]
        head = snake['corps'][0]
        idx_dir = snake['index_direction']
        
        # 8 directions absolues dans l'ordre horaire à partir de DROITE
        DIR_8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        
        # Trouver l'indice de départ dans DIR_8 correspondant à la direction du serpent
        # Les directions cardinales sont aux indices 0, 2, 4, 6 de DIR_8
        start_idx = idx_dir * 2
        
        state = []
        
        for i in range(8):
            dx, dy = DIR_8[(start_idx + i) % 8]
            
            dist_wall = 0.0
            dist_food = 0.0
            dist_body = 0.0
            
            found_food = False
            found_body = False
            
            k = 1
            while True:
                nx, ny = head[0] + k * dx, head[1] + k * dy
                
                # Détection mur (sortie de grille)
                if not (0 <= nx < self.largeur and 0 <= ny < self.hauteur):
                    dist_wall = 1.0 / k
                    break
                    
                # Détection pomme
                if not found_food and (nx, ny) in self.apples:
                    dist_food = 1.0 / k
                    found_food = True
                    
                # Détection corps d'un serpent quelconque
                if not found_body:
                    for sid, s in self.snakes.items():
                        if (nx, ny) in s['corps']:
                            dist_body = 1.0 / k
                            found_body = True
                            break
                k += 1
                
            state.extend([dist_wall, dist_food, dist_body])
            
        return np.array(state, dtype=np.float32)

    # =============================================================================
    # BOUCLE SYNCHRONE — STEP DE SIMULATION
    # =============================================================================
    def step(self, actions):
        """
        Fait avancer d'une frame tous les serpents de manière synchrone.
        
        Paramètres :
            actions (dict) : mapping snake_id -> action (0, 1, 2)
            
        Retourne :
            dead_snakes (set) : ID des serpents morts pendant ce tick
            rewards (dict)    : récompenses obtenues pour chaque serpent
        """
        dead_snakes = set()
        rewards = {}
        
        # 1. Calculer les nouvelles têtes pour chaque serpent
        nouvelles_tetes = {}
        for sid, action in actions.items():
            snake = self.snakes[sid]
            idx_dir = snake['index_direction']
            
            if action == 1:    # Gauche
                idx_dir = (idx_dir - 1) % 4
            elif action == 2:  # Droite
                idx_dir = (idx_dir + 1) % 4
                
            snake['index_direction'] = idx_dir
            snake['direction'] = SENS_HORAIRE[idx_dir]
            
            head = snake['corps'][0]
            nouvelles_tetes[sid] = (head[0] + snake['direction'][0], head[1] + snake['direction'][1])
            
        # 2. Détection des collisions simultanées
        for sid, new_head in nouvelles_tetes.items():
            snake = self.snakes[sid]
            
            # Collision 1 : Murs
            if not (0 <= new_head[0] < self.largeur and 0 <= new_head[1] < self.hauteur):
                dead_snakes.add(sid)
                continue
                
            # Collision 2 : Têtes contre Têtes (double mort instantanée)
            collision_tetes = False
            for other_id, other_head in nouvelles_tetes.items():
                if other_id != sid and new_head == other_head:
                    dead_snakes.add(sid)
                    collision_tetes = True
                    break
            if collision_tetes:
                continue
                
            # Collision 3 : Tête contre Corps d'un serpent
            # Règle physique : la queue s'en va au même moment si le serpent ne mange pas.
            collision_corps = False
            for other_id, other_s in self.snakes.items():
                # Déterminer si l'autre serpent va manger (pour savoir si sa queue libère la case)
                autre_va_manger = (nouvelles_tetes[other_id] in self.apples)
                corps_a_verifier = other_s['corps'] if autre_va_manger else other_s['corps'][:-1]
                
                if new_head in corps_a_verifier:
                    collision_corps = True
                    break
            if collision_corps:
                dead_snakes.add(sid)
                
        # 3. Traiter le mouvement, la nourriture et les récompenses
        pommes_mangees = []
        
        for sid, snake in self.snakes.items():
            if sid in dead_snakes:
                rewards[sid] = -10.0
                continue
                
            new_head = nouvelles_tetes[sid]
            old_head = snake['corps'][0]
            
            # Mesure de distance pour le reward shaping (Standard & Profond)
            dist_avant = float('inf')
            if self.apples:
                closest_apple = min(self.apples, key=lambda a: abs(a[0] - old_head[0]) + abs(a[1] - old_head[1]))
                dist_avant = abs(closest_apple[0] - old_head[0]) + abs(closest_apple[1] - old_head[1])
                
            # Ajouter la nouvelle tête
            snake['corps'].insert(0, new_head)
            
            # Est-ce qu'on mange une pomme ?
            if new_head in self.apples:
                pommes_mangees.append(new_head)
                if new_head in self.apples:
                    self.apples.remove(new_head)
                    
                snake['score_courant'] += 1
                snake['total_pommes_historique'] += 1
                snake['max_longueur'] = max(snake['max_longueur'], len(snake['corps']))
                snake['steps_sans_manger'] = 0
                rewards[sid] = 10.0
            else:
                # Retirer la queue
                snake['corps'].pop()
                snake['steps_sans_manger'] += 1
                
                # Système anti-boucle (mort par inanition)
                # Le seuil est proportionnel à la longueur pour permettre aux grands serpents de chercher la nourriture
                limite_inanition = max(200, 50 * len(snake['corps']))
                if snake['steps_sans_manger'] > limite_inanition:
                    dead_snakes.add(sid)
                    rewards[sid] = -10.0
                    continue
                    
                # Récompense de rapprochement (Reward shaping)
                if self.apples:
                    closest_apple = min(self.apples, key=lambda a: abs(a[0] - new_head[0]) + abs(a[1] - new_head[1]))
                    dist_apres = abs(closest_apple[0] - new_head[0]) + abs(closest_apple[1] - new_head[1])
                    if dist_apres < dist_avant:
                        rewards[sid] = 1.0
                    elif dist_apres > dist_avant:
                        rewards[sid] = -1.0
                    else:
                        rewards[sid] = 0.0
                else:
                    rewards[sid] = 0.0
                    
        # 4. Gérer les morts et les respawns
        for sid in dead_snakes:
            self.snakes[sid]['nb_respawns'] += 1
            self.snakes[sid]['score_courant'] = 0
            self.respawn_serpent(sid)
            
        # 5. Remplacer les pommes mangées
        for _ in range(len(pommes_mangees)):
            self.placer_nouvelle_pomme()
            
        return dead_snakes, rewards
