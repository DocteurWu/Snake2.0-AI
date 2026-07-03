# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Entraînement DQN découplé, Batché & Mode Graphe Plein Écran
=============================================================================
 Sépare complètement l'entraînement en arrière-plan (vitesse maximale) 
 de l'affichage visuel des 7 serpents de démo (vitesse humaine).
 
 Ajout du mode "Plein Écran Graphe" interactif avec illumination du dernier score.
=============================================================================
"""

import os
import sys
import time
import threading
import pygame
import numpy as np

from config import (
    TAILLE_GRILLE, TAILLE_CASE, VITESSE_JEU,
    NOIR, BLANC, ROUGE, VERT_FONCE, VERT_CLAIR, GRIS, BLEU,
    DROITE, GAUCHE, HAUT, BAS, SENS_HORAIRE,
    TAILLE_ETAT, NB_ACTIONS, NOM_MODELE_DQN
)
from game import SnakeGame
from agent_dqn import AgentDQN

# Dimensions
LARGEUR_SOUS_JEU = TAILLE_GRILLE * TAILLE_CASE # 400 px
HAUTEUR_SOUS_JEU = TAILLE_GRILLE * TAILLE_CASE # 400 px
NB_COLONNES = 4
NB_LIGNES = 2

LARGEUR_FENETRE = LARGEUR_SOUS_JEU * NB_COLONNES # 1600 px
HAUTEUR_FENETRE = HAUTEUR_SOUS_JEU * NB_LIGNES   # 800 px

lock_modele = threading.Lock()

class SubSnakeGame(SnakeGame):
    def __init__(self, surface):
        self.mode_graphique = True
        self.largeur = TAILLE_GRILLE
        self.hauteur = TAILLE_GRILLE
        self.surface = surface
        self.pygame = pygame
        self.police = pygame.font.SysFont('arial', 14)
        self.vitesse = 0
        self.reset()

    def render(self, id_jeu=0):
        self.surface.fill(NOIR)
        for x in range(0, self.largeur * TAILLE_CASE, TAILLE_CASE):
            pygame.draw.line(self.surface, GRIS, (x, 0), (x, self.hauteur * TAILLE_CASE))
        for y in range(0, self.hauteur * TAILLE_CASE, TAILLE_CASE):
            pygame.draw.line(self.surface, GRIS, (0, y), (self.largeur * TAILLE_CASE, y))
            
        fx, fy = self.nourriture
        pygame.draw.rect(self.surface, ROUGE, (
            fx * TAILLE_CASE + 2, fy * TAILLE_CASE + 2,
            TAILLE_CASE - 4, TAILLE_CASE - 4
        ), border_radius=5)
        
        for i, (sx, sy) in enumerate(self.corps):
            couleur = VERT_CLAIR if i == 0 else VERT_FONCE
            pygame.draw.rect(self.surface, couleur, (
                sx * TAILLE_CASE + 1, sy * TAILLE_CASE + 1,
                TAILLE_CASE - 2, TAILLE_CASE - 2
            ), border_radius=3)
            
        txt = self.police.render(f"IA #{id_jeu} | Score: {self.score}", True, BLANC)
        self.surface.blit(txt, (5, 5))


class Slider:
    def __init__(self, x, y, largeur, hauteur, val_min, val_max, val_init, titre="Vitesse"):
        self.rect = pygame.Rect(x, y, largeur, hauteur)
        self.min = val_min
        self.max = val_max
        self.valeur = val_init
        self.bouton_rect = pygame.Rect(x, y - 5, 10, hauteur + 10)
        self.titre = titre
        self.update_bouton_pos()
        self.drag = False
        self.mode_max = False

    def update_bouton_pos(self):
        ratio = (self.valeur - self.min) / (self.max - self.min)
        self.bouton_rect.x = self.rect.x + int(ratio * (self.rect.width - self.bouton_rect.width))

    def draw(self, surface):
        police = pygame.font.SysFont('arial', 13)
        pygame.draw.rect(surface, GRIS, self.rect, border_radius=3)
        
        largeur_remplissage = self.bouton_rect.centerx - self.rect.x
        if largeur_remplissage > 0:
            remplissage_rect = pygame.Rect(self.rect.x, self.rect.y, largeur_remplissage, self.rect.height)
            pygame.draw.rect(surface, BLEU, remplissage_rect, border_radius=3)

        couleur_curseur = BLANC if not self.drag else VERT_CLAIR
        pygame.draw.rect(surface, couleur_curseur, self.bouton_rect, border_radius=2)
        
        val_txt = "MAX (Débridé)" if self.mode_max else f"{int(self.valeur)} FPS"
        txt = police.render(f"{self.titre} : {val_txt}", True, BLANC)
        surface.blit(txt, (self.rect.x, self.rect.y - 18))

    def handle_event(self, event, offset_x, offset_y):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pos = (event.pos[0] - offset_x, event.pos[1] - offset_y)
                if self.bouton_rect.collidepoint(pos) or self.rect.collidepoint(pos):
                    self.drag = True
                    self.mode_max = False
                    self.update_valeur(pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.drag = False
        elif event.type == pygame.MOUSEMOTION:
            if self.drag:
                pos_x = event.pos[0] - offset_x
                self.update_valeur(pos_x)

    def update_valeur(self, pos_x):
        x_relatif = max(self.rect.x, min(pos_x, self.rect.x + self.rect.width - self.bouton_rect.width))
        ratio = (x_relatif - self.rect.x) / (self.rect.width - self.bouton_rect.width)
        self.valeur = self.min + ratio * (self.max - self.min)
        
        if self.valeur >= self.max - 20:
            self.mode_max = True
        else:
            self.mode_max = False
            
        self.update_bouton_pos()


# Variables partagées
scores_train = []
moyennes_train = []
en_pause = False
en_cours = True
vitesse_reelle_bg = 0
dernier_score_temps = 0 # Timestamp pour l'effet de flash à chaque mort d'arrière-plan
mode_plein_ecran_graphe = True # NOUVEAU: Par défaut, le graphe est en plein écran interactif

def thread_entrainement_background(agent, slider_bg):
    global scores_train, moyennes_train, en_pause, en_cours, vitesse_reelle_bg, dernier_score_temps
    
    nb_env = 20
    jeux_invisibles = [SnakeGame(mode_graphique=False) for _ in range(nb_env)]
    etats = [jeu.reset() for jeu in jeux_invisibles]
    liste_scores_recents = []
    
    step_compteur = 0
    dernier_temps = time.time()
    steps_mesures = 0
    
    while en_cours:
        if en_pause:
            time.sleep(0.1)
            continue
            
        fps_bg = 0 if slider_bg.mode_max else int(slider_bg.valeur)
        temps_debut_step = time.time()

        with lock_modele:
            actions = agent.choisir_actions_batch(etats, entrainement=True)

        for idx in range(nb_env):
            action = actions[idx]
            etat_actuel = etats[idx]
            
            etat_suivant, recompense, termine, score = jeux_invisibles[idx].step(action)
            agent.memoriser(etat_actuel, action, recompense, etat_suivant, termine)
            
            etats[idx] = etat_suivant
            steps_mesures += 1
            
            step_compteur += 1
            if step_compteur % 4 == 0:
                with lock_modele:
                    agent.entrainer()
            
            if termine:
                scores_train.append(score)
                liste_scores_recents.append(score)
                if len(liste_scores_recents) > 100:
                    liste_scores_recents.pop(0)
                moyennes_train.append(sum(liste_scores_recents) / len(liste_scores_recents))
                
                # Déclencher le flash lumineux sur le graphe
                dernier_score_temps = time.time()
                
                etats[idx] = jeux_invisibles[idx].reset()
                with lock_modele:
                    agent.fin_episode()
                
                if len(scores_train) % 100 == 0:
                    with lock_modele:
                        agent.sauvegarder()
                        
        temps_actuel = time.time()
        if temps_actuel - dernier_temps >= 1.0:
            vitesse_reelle_bg = int(steps_mesures / (temps_actuel - dernier_temps))
            steps_mesures = 0
            dernier_temps = temps_actuel

        if fps_bg > 0:
            temps_ecoule = time.time() - temps_debut_step
            temps_attente = (1.0 / fps_bg) - temps_ecoule
            if temps_attente > 0:
                time.sleep(temps_attente)


def dessiner_graphe_complet(surface, scores, moyennes, slider_bg, slider_vis, large, haut):
    """Dessine le graphe sur une largeur et hauteur données (s'adapte au mode démo ou plein écran)."""
    surface.fill((20, 20, 20))
    pygame.draw.rect(surface, BLEU, (0, 0, large, haut), 3)
    
    # Rendu des sliders décalés dynamiquement en fonction de la taille
    slider_bg.draw(surface)
    slider_vis.draw(surface)
    
    police = pygame.font.SysFont('arial', 14)
    police_titre = pygame.font.SysFont('arial', 18, bold=True)
    
    # Titre principal
    mode_nom = "MODE ENDLESS (PLEIN ÉCRAN GRAPHIC)" if mode_plein_ecran_graphe else "MODE DÉMONSTRATION 7 AGENTS"
    txt_titre = police_titre.render(mode_nom, True, BLANC)
    surface.blit(txt_titre, (30, 15))

    # Boutons d'interaction
    txt_bouton = "[CLIQUEZ ICI POUR CHANGER DE MODE]"
    pygame.draw.rect(surface, BLEU, (300, 12, 320, 26), border_radius=4)
    surface.blit(police.render(txt_bouton, True, BLANC), (310, 16))
    
    # Statut
    statut_txt = "STATUT : EN PAUSE [ESPACE]" if en_pause else f"VITESSE DE CALCUL (Background) : {vitesse_reelle_bg} steps/s"
    couleur_statut = ROUGE if en_pause else VERT_CLAIR
    surface.blit(police.render(statut_txt, True, couleur_statut), (30, 120))
    
    # Aire du graphe
    margin_x = 60
    margin_y = 50
    g_w = large - margin_x - 40
    g_h = haut - margin_y - 190
    
    base_y = haut - margin_y
    top_y = 170
    
    # Axes
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (large - 40, base_y), 2)
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (margin_x, top_y), 2)
    
    if len(scores) < 2:
        txt = police.render("En attente des premières parties d'arrière-plan...", True, GRIS)
        surface.blit(txt, (large // 2 - 150, top_y + g_h // 2))
        return
        
    max_score = max(max(scores), 1)
    nb_pts = len(scores)
    
    pts_scores = []
    pts_moyennes = []
    
    # Réduction dynamique des points affichés pour garder le tracé fluide
    pas = max(1, nb_pts // 800)
    indices = list(range(0, nb_pts, pas))
    if indices[-1] != nb_pts - 1:
        indices.append(nb_pts - 1)
        
    for idx, i in enumerate(indices):
        x = margin_x + int((idx / (len(indices) - 1)) * g_w)
        y_s = base_y - int((scores[i] / max_score) * g_h)
        pts_scores.append((x, y_s))
        
        y_m = base_y - int((moyennes[i] / max_score) * g_h)
        pts_moyennes.append((x, y_m))
        
    # Dessiner les courbes
    if len(pts_scores) > 1:
        pygame.draw.lines(surface, (80, 80, 80), False, pts_scores, 1) # Brut
        pygame.draw.lines(surface, VERT_CLAIR, False, pts_moyennes, 2)  # Moyenne mobile
        
    # Effet d'illumination sur le DERNIER point
    dernier_pt_x = pts_scores[-1][0]
    dernier_pt_y = pts_scores[-1][1]
    
    # Animation de pulsation lumineuse (pendant 0.3s après un score)
    t_ecoule = time.time() - dernier_score_temps
    if t_ecoule < 0.3:
        rayon_flash = int(12 * (1.0 - t_ecoule / 0.3))
        # Halo bleu transparent
        surface_halo = pygame.Surface((rayon_flash*2, rayon_flash*2), pygame.SRCALPHA)
        pygame.draw.circle(surface_halo, (0, 200, 255, 120), (rayon_flash, rayon_flash), rayon_flash)
        surface.blit(surface_halo, (dernier_pt_x - rayon_flash, dernier_pt_y - rayon_flash))
    
    # Cercle brillant principal
    pygame.draw.circle(surface, BLANC, (dernier_pt_x, dernier_pt_y), 5)
    pygame.draw.circle(surface, ROUGE, (dernier_pt_x, dernier_pt_y), 3)

    # Statistiques affichées à droite
    lbl_max = police.render(f"Meilleur score : {max_score}", True, ROUGE)
    lbl_avg = police.render(f"Moyenne (100) : {moyennes[-1]:.1f}", True, VERT_CLAIR)
    lbl_partie = police.render(f"Épisodes joués : {len(scores)}", True, BLANC)
    
    surface.blit(lbl_max, (large - 180, 15))
    surface.blit(lbl_avg, (large - 180, 35))
    surface.blit(lbl_partie, (large - 180, 55))


def main():
    global en_pause, en_cours, mode_plein_ecran_graphe
    pygame.init()
    fenetre_globale = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
    pygame.display.set_caption("🐍 Snake IA — Mode Endless Graphe Lumineux & Rendu Décoquillé")
    horloge = pygame.time.Clock()
    
    # 8 sous-surfaces de base
    surfaces_grille = []
    for lig in range(NB_LIGNES):
        for col in range(NB_COLONNES):
            rect = pygame.Rect(col * LARGEUR_SOUS_JEU, lig * HAUTEUR_SOUS_JEU, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU)
            surfaces_grille.append(fenetre_globale.subsurface(rect))
            
    # Surface Plein Écran pour le Graphe Geste (Endless)
    rect_plein_ecran = pygame.Rect(0, 0, LARGEUR_FENETRE, HAUTEUR_FENETRE)
    surface_plein_ecran = fenetre_globale.subsurface(rect_plein_ecran)
    
    # Sliders
    # Train BG : curseur de vitesse background
    slider_bg = Slider(30, 55, 240, 10, 10, 2000, 2000, titre="Vitesse Entraînement")
    slider_bg.mode_max = True # Par défaut
    
    # Rendu Visuel : vitesse affichage démo (utilisé en mode démo)
    slider_vis = Slider(30, 95, 240, 10, 5, 120, 20, titre="Vitesse Démo")
    
    # 7 instances de démonstration
    jeux_demo = [SubSnakeGame(surfaces_grille[i]) for i in range(7)]
    etats_demo = [jeu.reset() for jeu in jeux_demo]
    
    agent = AgentDQN()
    if os.path.exists(os.path.join("models", NOM_MODELE_DQN)):
        try:
            agent.charger()
        except Exception:
            pass

    # Thread d'arrière-plan
    thread_bg = threading.Thread(
        target=thread_entrainement_background, 
        args=(agent, slider_bg),
        daemon=True
    )
    thread_bg.start()
    
    print("\nSimulation et entraînements démarrés !")
    
    while en_cours:
        fps_visual = int(slider_vis.valeur)
        
        # Déterminer les offsets de clics de souris en fonction du mode actif
        if mode_plein_ecran_graphe:
            # En plein écran, le graphe est à la coordonnée (0,0) de la fenêtre
            offset_x, offset_y = 0, 0
        else:
            # En mode démo, le graphe est en case 8 (colonne 3, ligne 1)
            offset_x = 3 * LARGEUR_SOUS_JEU
            offset_y = 1 * HAUTEUR_SOUS_JEU

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                en_cours = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    en_cours = False
                elif event.key == pygame.K_SPACE:
                    en_pause = not en_pause
                elif event.key == pygame.K_s:
                    with lock_modele:
                        agent.sauvegarder()
            
            # Relayer événements souris aux sliders avec offset dynamique
            slider_bg.handle_event(event, offset_x, offset_y)
            slider_vis.handle_event(event, offset_x, offset_y)
            
            # Clic pour basculer de mode (Bouton rapide en haut)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = (event.pos[0] - offset_x, event.pos[1] - offset_y)
                # Si clic sur la zone bouton "Changer de mode"
                if 300 <= pos[0] <= 620 and 12 <= pos[1] <= 38:
                    mode_plein_ecran_graphe = not mode_plein_ecran_graphe
                    # Effacer l'écran pour éviter les résidus graphiques de la grille
                    fenetre_globale.fill(NOIR)

        # 1. Rendre le jeu visuel (uniquement en mode Démonstration)
        if not mode_plein_ecran_graphe and not en_pause:
            for idx, jeu in enumerate(jeux_demo):
                with lock_modele:
                    action = agent.choisir_action(etats_demo[idx], entrainement=False)
                etat_suivant, _, termine, _ = jeu.step(action)
                jeu.render(id_jeu=idx+1)
                etats_demo[idx] = etat_suivant
                if termine:
                    etats_demo[idx] = jeu.reset()

        # 2. Rendre le graphe interactif
        if mode_plein_ecran_graphe:
            dessiner_graphe_complet(surface_plein_ecran, scores_train, moyennes_train, slider_bg, slider_vis, LARGEUR_FENETRE, HAUTEUR_FENETRE)
        else:
            dessiner_graphe_complet(surfaces_grille[7], scores_train, moyennes_train, slider_bg, slider_vis, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU)
            
        pygame.display.flip()
        
        # Réguler le rendu visuel
        if not mode_plein_ecran_graphe:
            horloge.tick(fps_visual)
        else:
            # En plein écran graphe, on limite à 60 FPS l'affichage de l'interface graphique pour reposer le GPU
            horloge.tick(60)
            
    en_cours = False
    thread_bg.join(timeout=1.0)
    with lock_modele:
        agent.sauvegarder()
    pygame.quit()
    print("Entraînement terminé.")

if __name__ == "__main__":
    main()
