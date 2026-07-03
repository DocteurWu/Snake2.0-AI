# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA MARL — Arène de Simulation Compétitive
=============================================================================
 Script principal : lance l'arène de 8 serpents avec interface Pygame complète.
 Contrôles :
   ESPACE : Pause / Reprise
   HAUT   : Augmenter la vitesse (FPS)
   BAS    : Diminuer la vitesse (FPS)
=============================================================================
"""

import os
import sys
import time
import pygame

from marl_game import MARLGame, DROITE, BAS, GAUCHE, HAUT
from marl_agents import (
    DQNAgent, AStarAgent, EconomistAgent, StrategistAgent
)

# Dimensions
TAILLE_CASE = 15
LARGEUR_GRILLE = 40
HAUTEUR_GRILLE = 40

LARGEUR_JEU = LARGEUR_GRILLE * TAILLE_CASE  # 600 px
LARGEUR_HUD = 400                           # 400 px
HAUTEUR_FENETRE = HAUTEUR_GRILLE * TAILLE_CASE # 600 px
LARGEUR_FENETRE = LARGEUR_JEU + LARGEUR_HUD  # 1000 px

# Couleurs (Aesthetics premium)
NOIR_ARENE = (10, 10, 10)
GRIS_GRILLE = (25, 25, 25)
NOIR_HUD = (20, 20, 20)
GRIS_BORDURE = (40, 40, 40)
BLANC = (240, 240, 240)
GRIS_TEXTE = (160, 160, 160)
OR_TITRE = (230, 185, 30)
ROUGE_POMME = (230, 50, 50)

# Couleurs uniques distinctes pour les 8 serpents
COULEURS_SERPENTS = [
    (80, 180, 255),  # 1. Standard : Bleu clair
    (255, 60, 60),   # 2. Profond : Rouge vif
    (200, 60, 200),  # 3. Raycast : Magenta / Violet
    (50, 240, 50),   # 4. Mathématicien (A*) : Vert fluo
    (240, 220, 40),  # 5. Psychologue (Minimax) : Jaune
    (255, 130, 0),   # 6. Économiste (Influence) : Orange
    (40, 220, 220),  # 7. Stratège (Game Theory) : Cyan
    (110, 110, 255)  # 8. Historien (KNN) : Bleu indigo
]

def main():
    print("=" * 70)
    print("        DEMARRAGE DE L'ARENE COMPETITIVE SNAKE MARL")
    print("=" * 70)
    print("Description des concurrents :")
    print("  - Mathématicien (A*) : Calcule le chemin le plus court vers la pomme la plus")
    print("                         proche en évitant les obstacles. Fallback de survie si bloqué.")
    print("  - Élite (DQN Sélectif) : Réseau profond apprenant en continu de ses données")
    print("                           et des transitions du meilleur serpent actuel de la session.")
    print("  - Économiste (Influence) : Évalue le potentiel des cases adjacentes (attraction pomme +1,")
    print("                             répulsion obstacles -5 et têtes adverses -10).")
    print("  - Stratège (GameTheory) : Anticipe la trajectoire adverse sur 2 pas pour couper la route")
    print("                            et provoquer un crash direct (Kill).")
    print("  - Collectif (DQN Partagé) : Réseau profond apprenant en continu des trajectoires 11D")
    print("                              de TOUS les 8 serpents de l'arène dans son Replay Buffer.")
    print("=" * 70)
    
    # 1. Initialiser Pygame
    pygame.init()
    fenetre = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
    pygame.display.set_caption("Snake IA MARL - Arène Compétitive")
    horloge = pygame.time.Clock()
    
    # Polices
    police_titre = pygame.font.SysFont('arial', 18, bold=True)
    police_section = pygame.font.SysFont('arial', 14, bold=True)
    police_normal = pygame.font.SysFont('arial', 12)
    police_normal_bold = pygame.font.SysFont('arial', 12, bold=True)
    
    # 2. Créer le moteur de jeu
    jeu = MARLGame(LARGEUR_GRILLE, HAUTEUR_GRILLE, nb_pommes=8)
    
    # Configurer les 8 serpents
    jeu.ajouter_serpent(0, "Standard (DQN)", COULEURS_SERPENTS[0])
    jeu.ajouter_serpent(1, "Profond (DQN)", COULEURS_SERPENTS[1])
    jeu.ajouter_serpent(2, "Raycast (DQN)", COULEURS_SERPENTS[2])
    jeu.ajouter_serpent(3, "Mathématicien (A*)", COULEURS_SERPENTS[3])
    jeu.ajouter_serpent(4, "Élite (DQN Sélectif)", COULEURS_SERPENTS[4])
    jeu.ajouter_serpent(5, "Économiste (Influence)", COULEURS_SERPENTS[5])
    jeu.ajouter_serpent(6, "Stratège (GameTheory)", COULEURS_SERPENTS[6])
    jeu.ajouter_serpent(7, "Collectif (DQN Partagé)", COULEURS_SERPENTS[7])
    
    # Placer les pommes
    jeu.initialiser_pommes()
    
    # 3. Initialiser les 8 Agents
    print("\n[Initialisation des Agents...]")
    agents = {
        0: DQNAgent("Standard", taille_entree=11, couches_cachees=[256, 128]),
        1: DQNAgent("Profond", taille_entree=11, couches_cachees=[256, 128, 64, 32], epsilon_decay=0.9995),
        2: DQNAgent("Raycast", taille_entree=24, couches_cachees=[256, 128]),
        3: AStarAgent(),
        4: DQNAgent("Elite", taille_entree=11, couches_cachees=[256, 128, 64, 32], epsilon_decay=0.9995),
        5: EconomistAgent(),
        6: StrategistAgent(),
        7: DQNAgent("Collectif", taille_entree=11, couches_cachees=[256, 128, 64, 32], epsilon_decay=0.9995)
    }
    
    # Charger les checkpoints pour les 5 DQN
    for sid in [0, 1, 2, 4, 7]:
        if agents[sid].charger_si_existe():
            # Restaurer les statistiques historiques dans l'arène
            s = jeu.snakes[sid]
            s['nb_respawns'] = agents[sid].nb_episodes
            s['total_pommes_historique'] = int(agents[sid].meilleure_moyenne * agents[sid].nb_episodes)
            # Reconstituer l'historique des 10 dernières vies avec la moyenne chargée
            nb_vies_chargees = min(10, agents[sid].nb_episodes)
            s['historique_scores'].extend([int(agents[sid].meilleure_moyenne)] * nb_vies_chargees)

    # Variables de contrôle de simulation
    vitesses_fps = [5, 10, 15, 25, 40, 60, 120, 250, 500, 0] # 0 = Max / Illimité
    idx_vitesse = 3  # Par défaut 25 FPS
    en_pause = False
    etape = 0
    en_cours = True
    
    print("\n🔑 Commandes Clavier :")
    print("   [ESPACE] : Mettre en pause / Reprendre")
    print("   [HAUT]   : Accélérer le jeu")
    print("   [BAS]    : Ralentir le jeu")
    print("   [R]      : Réinitialiser les scores historiques")
    print("=" * 60)
    
    while en_cours:
        # Gérer les entrées clavier
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                en_cours = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    en_pause = not en_pause
                    print(f"[Arena] Pause : {en_pause}")
                elif event.key == pygame.K_UP:
                    idx_vitesse = min(len(vitesses_fps) - 1, idx_vitesse + 1)
                    v_txt = "Illimitée" if vitesses_fps[idx_vitesse] == 0 else f"{vitesses_fps[idx_vitesse]} FPS"
                    print(f"[Arena] Vitesse : {v_txt}")
                elif event.key == pygame.K_DOWN:
                    idx_vitesse = max(0, idx_vitesse - 1)
                    v_txt = "Illimitée" if vitesses_fps[idx_vitesse] == 0 else f"{vitesses_fps[idx_vitesse]} FPS"
                    print(f"[Arena] Vitesse : {v_txt}")
                elif event.key == pygame.K_r:
                    # Réinitialiser statistiques
                    for s in jeu.snakes.values():
                        s['nb_respawns'] = 0
                        s['total_pommes_historique'] = 0
                        s['score_courant'] = 0
                        s['max_longueur'] = len(s['corps'])
                    print("[Arena] Statistiques réinitialisées.")
                    
        if not en_pause:
            etape += 1
            
            # --- 1. Collecter les états initiaux pour chaque serpent ---
            etats_courants = {}
            for sid, s in jeu.snakes.items():
                if sid == 0 or sid == 1 or sid == 7:
                    # États 11D
                    etats_courants[sid] = jeu.get_11d_state(sid)
                elif sid == 2:
                    # Raycast 24D
                    etats_courants[sid] = jeu.get_raycast_state(sid)
            
            # Enregistrer l'état 11D avant mouvement pour tous les serpents (requis pour les buffers partagés)
            etats_11d_tous = {}
            for sid in range(8):
                etats_11d_tous[sid] = jeu.get_11d_state(sid)
                
            for sid, s in jeu.snakes.items():
                if sid in [0, 1, 4, 7]:
                    # États 11D
                    etats_courants[sid] = etats_11d_tous[sid]
                elif sid == 2:
                    # Raycast 24D
                    etats_courants[sid] = jeu.get_raycast_state(sid)
            
            # --- 2. Choisir l'action pour chaque agent ---
            actions = {}
            for sid in range(8):
                if sid in [0, 1, 4, 7]:
                    actions[sid] = agents[sid].choisir_action(etats_courants[sid])
                elif sid == 2:
                    actions[sid] = agents[sid].choisir_action(etats_courants[sid])
                else:
                    # Agents algorithmiques
                    actions[sid] = agents[sid].choisir_action(jeu, sid)
                    
            # --- 3. Avancer la simulation d'un tick synchrone ---
            morts, recompenses = jeu.step(actions)
            
            # --- 4. Entraîner en continu les DQN ---
            # Récupérer les états suivants après mouvement en 11D pour tout le monde
            next_states_11d = {}
            for sid in range(8):
                next_states_11d[sid] = jeu.get_11d_state(sid)
            next_state_ray = jeu.get_raycast_state(2)
            
            # Serpent 1 (Standard)
            done1 = (0 in morts)
            agents[0].memoriser(etats_courants[0], actions[0], recompenses[0], next_states_11d[0], done1)
            for _ in range(4):
                agents[0].entrainer()
            if done1:
                agents[0].fin_episode()
                moy_std = jeu.obtenir_moyenne_10_vies(0)
                agents[0].sauvegarder_si_meilleur(moy_std)
                
            # Serpent 2 (Profond)
            done2 = (1 in morts)
            agents[1].memoriser(etats_courants[1], actions[1], recompenses[1], next_states_11d[1], done2)
            for _ in range(4):
                agents[1].entrainer()
            if done2:
                agents[1].fin_episode()
                moy_prof = jeu.obtenir_moyenne_10_vies(1)
                agents[1].sauvegarder_si_meilleur(moy_prof)
                
            # Serpent 3 (Raycast)
            done3 = (2 in morts)
            agents[2].memoriser(etats_courants[2], actions[2], recompenses[2], next_state_ray, done3)
            for _ in range(4):
                agents[2].entrainer()
            if done3:
                agents[2].fin_episode()
                moy_ray = jeu.obtenir_moyenne_10_vies(2)
                agents[2].sauvegarder_si_meilleur(moy_ray)
                
            # Trouver le meilleur serpent actuel du classement (basé sur la moyenne des 10 dernières vies)
            meilleur_sid = None
            meilleure_moy = -1.0
            for sid, s in jeu.snakes.items():
                moy_s = jeu.obtenir_moyenne_10_vies(sid)
                if moy_s > meilleure_moy:
                    meilleure_moy = moy_s
                    meilleur_sid = sid
            
            # Serpent 5 (Élite - DQN Sélectif)
            # Apprend de ses données ET de celles du meilleur serpent de la session
            done4 = (4 in morts)
            agents[4].memoriser(etats_courants[4], actions[4], recompenses[4], next_states_11d[4], done4)
            if meilleur_sid is not None and meilleur_sid != 4:
                done_best = (meilleur_sid in morts)
                agents[4].memoriser(etats_11d_tous[meilleur_sid], actions[meilleur_sid], recompenses[meilleur_sid], next_states_11d[meilleur_sid], done_best)
            
            for _ in range(4):
                agents[4].entrainer()
            if done4:
                agents[4].fin_episode()
                moy_el = jeu.obtenir_moyenne_10_vies(4)
                agents[4].sauvegarder_si_meilleur(moy_el)
                
            # Serpent 8 (Collectif - DQN Partagé)
            # Apprend des transitions 11D de tous les 8 serpents de l'arène
            for sid in range(8):
                done_sid = (sid in morts)
                agents[7].memoriser(etats_11d_tous[sid], actions[sid], recompenses[sid], next_states_11d[sid], done_sid)
            
            for _ in range(4):
                agents[7].entrainer()
            done7 = (7 in morts)
            if done7:
                agents[7].fin_episode()
                moy_col = jeu.obtenir_moyenne_10_vies(7)
                agents[7].sauvegarder_si_meilleur(moy_col)

        # =============================================================================
        # RENDU GRAPHIQUE (Pygame)
        # =============================================================================
        fenetre.fill(NOIR_ARENE)
        
        # --- 1. Dessiner la grille ---
        for x in range(0, LARGEUR_JEU, TAILLE_CASE):
            pygame.draw.line(fenetre, GRIS_GRILLE, (x, 0), (x, HAUTEUR_FENETRE))
        for y in range(0, HAUTEUR_FENETRE, TAILLE_CASE):
            pygame.draw.line(fenetre, GRIS_GRILLE, (0, y), (LARGEUR_JEU, y))
            
        # --- 2. Dessiner les pommes ---
        for ax, ay in jeu.apples:
            pygame.draw.rect(fenetre, ROUGE_POMME, (
                ax * TAILLE_CASE + 2, ay * TAILLE_CASE + 2,
                TAILLE_CASE - 4, TAILLE_CASE - 4
            ), border_radius=6)
            
        # --- 3. Dessiner les serpents ---
        for sid, s in jeu.snakes.items():
            # Dessiner le corps (légèrement plus petit pour détacher les segments)
            for idx, (sx, sy) in enumerate(s['corps']):
                is_head = (idx == 0)
                radius = 5 if is_head else 3
                offset = 1 if is_head else 2
                size = TAILLE_CASE - 2 * offset
                
                # Couleur plus sombre pour le corps
                color = s['couleur']
                if not is_head:
                    # Assombrir le corps de 30%
                    color = tuple(max(0, int(c * 0.7)) for c in color)
                    
                pygame.draw.rect(fenetre, color, (
                    sx * TAILLE_CASE + offset, sy * TAILLE_CASE + offset,
                    size, size
                ), border_radius=radius)
                
                # Si c'est la tête, dessiner deux petits yeux pour le style !
                if is_head:
                    head_dir = s['direction']
                    # Positions d'yeux simplistes
                    eye_color = (255, 255, 255)
                    eye_r = 2
                    cx = sx * TAILLE_CASE + TAILLE_CASE // 2
                    cy = sy * TAILLE_CASE + TAILLE_CASE // 2
                    
                    if head_dir == (1, 0) or head_dir == (-1, 0): # Horizontal
                        pygame.draw.circle(fenetre, eye_color, (cx, cy - 3), eye_r)
                        pygame.draw.circle(fenetre, eye_color, (cx, cy + 3), eye_r)
                    else: # Vertical
                        pygame.draw.circle(fenetre, eye_color, (cx - 3, cy), eye_r)
                        pygame.draw.circle(fenetre, eye_color, (cx + 3, cy), eye_r)

        # --- 4. Dessiner le HUD latéral ---
        hud_rect = pygame.Rect(LARGEUR_JEU, 0, LARGEUR_HUD, HAUTEUR_FENETRE)
        pygame.draw.rect(fenetre, NOIR_HUD, hud_rect)
        pygame.draw.line(fenetre, GRIS_BORDURE, (LARGEUR_JEU, 0), (LARGEUR_JEU, HAUTEUR_FENETRE), 2)
        
        # Titre HUD
        titre_txt = police_titre.render("ARENE COMPETITIVE MARL", True, OR_TITRE)
        fenetre.blit(titre_txt, (LARGEUR_JEU + 25, 20))
        
        # Informations générales
        fps_lim = vitesses_fps[idx_vitesse]
        fps_txt = "Illimité" if fps_lim == 0 else f"{fps_lim} FPS"
        if en_pause:
            fps_txt += " (PAUSE)"
            
        info_txt = f"Tick : {etape}  |  Vitesse : {fps_txt}"
        info_surf = police_normal.render(info_txt, True, GRIS_TEXTE)
        fenetre.blit(info_surf, (LARGEUR_JEU + 15, 50))
        
        # Ligne de séparation
        pygame.draw.line(fenetre, GRIS_BORDURE, (LARGEUR_JEU + 10, 75), (LARGEUR_FENETRE - 10, 75), 1)
        
        # En-tête classement
        class_title = police_section.render("CLASSEMENT (Moyenne 10 Vies)", True, BLANC)
        fenetre.blit(class_title, (LARGEUR_JEU + 15, 85))
        
        stats_serpents = []
        for sid, s in jeu.snakes.items():
            morts = s['nb_respawns']
            moyenne = jeu.obtenir_moyenne_10_vies(sid)
            stats_serpents.append({
                'id': sid,
                'nom': s['nom'],
                'color': s['couleur'],
                'morts': morts,
                'max_len': s['max_longueur'],
                'moyenne': moyenne,
                'eps': agents[sid].epsilon if sid in [0, 1, 2, 4, 7] else None,
                'nb_donnees': agents[sid].total_transitions if sid in [0, 1, 2, 4, 7] else None
            })
            
        stats_serpents.sort(key=lambda x: (x['moyenne'], x['max_len']), reverse=True)
        
        # Afficher la liste triée
        y_offset = 120
        for i, stat in enumerate(stats_serpents):
            # 1. Numéro et indicateur couleur
            num_txt = police_normal_bold.render(f"{i+1}.", True, BLANC)
            fenetre.blit(num_txt, (LARGEUR_JEU + 15, y_offset))
            
            # Carré de couleur
            pygame.draw.rect(fenetre, stat['color'], (
                LARGEUR_JEU + 35, y_offset + 2, 10, 10
            ))
            
            # 2. Nom de l'architecture
            nom_txt = stat['nom']
            # Ajouter epsilon pour les DQN
            if stat['eps'] is not None:
                nom_txt += f" (ε={stat['eps']:.2f})"
                
            nom_surf = police_normal_bold.render(nom_txt, True, BLANC)
            fenetre.blit(nom_surf, (LARGEUR_JEU + 52, y_offset))
            
            # 3. Statistiques de performance
            stats_str = f"Moyenne: {stat['moyenne']:.2f}  |  Morts: {stat['morts']}  |  Record: {stat['max_len']}"
            if stat['nb_donnees'] is not None:
                stats_str += f"  |  Données: {stat['nb_donnees']}"
            stats_surf = police_normal.render(stats_str, True, GRIS_TEXTE)
            fenetre.blit(stats_surf, (LARGEUR_JEU + 35, y_offset + 16))
            
            y_offset += 48
            
        # Ligne de pied de page
        pygame.draw.line(fenetre, GRIS_BORDURE, (LARGEUR_JEU + 10, 520), (LARGEUR_FENETRE - 10, 520), 1)
        
        # Astuces clavier
        note1 = police_normal.render("[ESPACE] : Pause/Reprise", True, GRIS_TEXTE)
        note2 = police_normal.render("[HAUT/BAS] : Modifier vitesse (FPS)", True, GRIS_TEXTE)
        note3 = police_normal.render("[R] : Réinitialiser les statistiques", True, GRIS_TEXTE)
        fenetre.blit(note1, (LARGEUR_JEU + 15, 530))
        fenetre.blit(note2, (LARGEUR_JEU + 15, 550))
        fenetre.blit(note3, (LARGEUR_JEU + 15, 570))
        
        pygame.display.flip()
        
        # Limitation de vitesse (FPS)
        vitesse_actuelle = vitesses_fps[idx_vitesse]
        if vitesse_actuelle > 0 and not en_pause:
            horloge.tick(vitesse_actuelle)
        else:
            # Même en pause ou illimité, on fait un petit sleep pour ne pas brûler 100% du CPU
            if en_pause:
                pygame.time.wait(100)
            else:
                # Vitesse maximale, tick à 1000 FPS max pour reposer le thread
                horloge.tick(1000)
                
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
