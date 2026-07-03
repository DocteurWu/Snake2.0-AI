# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Entraînement DQN Multi-Agent Découplé (Double Curseurs)
=============================================================================
 Fait tourner 7 serpents en parallèle.
 Deux sliders distincts permettent de gérer :
   1. La vitesse de l'entraînement (nombre d'étapes calculées en arrière-plan)
   2. La vitesse de l'affichage à l'écran (FPS des serpents de démonstration)
 
 Contrôles :
   Clic souris : Ajuster les Sliders dans la 8ème case
   ESPACE      : Mettre l'entraînement en pause / reprise
   S           : Sauvegarder manuellement le modèle
   ECHAP       : Sauvegarder et quitter
=============================================================================
"""

import os
import sys
import time
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

# Dimensions de la fenêtre globale
LARGEUR_SOUS_JEU = TAILLE_GRILLE * TAILLE_CASE # 400 px
HAUTEUR_SOUS_JEU = TAILLE_GRILLE * TAILLE_CASE # 400 px
NB_COLONNES = 4
NB_LIGNES = 2

LARGEUR_FENETRE = LARGEUR_SOUS_JEU * NB_COLONNES # 1600 px
HAUTEUR_FENETRE = HAUTEUR_SOUS_JEU * NB_LIGNES   # 800 px

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
        # Grille
        for x in range(0, self.largeur * TAILLE_CASE, TAILLE_CASE):
            pygame.draw.line(self.surface, GRIS, (x, 0), (x, self.hauteur * TAILLE_CASE))
        for y in range(0, self.hauteur * TAILLE_CASE, TAILLE_CASE):
            pygame.draw.line(self.surface, GRIS, (0, y), (self.largeur * TAILLE_CASE, y))
            
        # Nourriture
        fx, fy = self.nourriture
        pygame.draw.rect(self.surface, ROUGE, (
            fx * TAILLE_CASE + 2, fy * TAILLE_CASE + 2,
            TAILLE_CASE - 4, TAILLE_CASE - 4
        ), border_radius=5)
        
        # Serpent
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
        self.titre = titre
        self.bouton_rect = pygame.Rect(x, y - 5, 10, hauteur + 10)
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
        
        val_txt = "MAX (No limit)" if self.mode_max else f"{int(self.valeur)}"
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
        
        if self.valeur >= self.max - (self.max - self.min) * 0.02:
            self.mode_max = True
        else:
            self.mode_max = False
            
        self.update_bouton_pos()


def dessiner_graphe_et_sliders(surface, scores, moyennes, slider_train, slider_fps):
    """Dessine le graphe d'évolution et les 2 sliders dans la case 8."""
    surface.fill((20, 20, 20))
    pygame.draw.rect(surface, BLEU, (0, 0, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU), 2)
    
    # Rendu des 2 Sliders
    slider_train.draw(surface)
    slider_fps.draw(surface)
    
    # Affichage du graphe
    margin_x, margin_y = 50, 40
    g_w = LARGEUR_SOUS_JEU - margin_x - 20
    g_h = HAUTEUR_SOUS_JEU - margin_y - 170 # Libère plus de place en haut
    
    base_y = HAUTEUR_SOUS_JEU - margin_y
    top_y = 160
    
    # Axes
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (LARGEUR_SOUS_JEU - 20, base_y), 1)
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (margin_x, top_y), 1)
    
    police = pygame.font.SysFont('arial', 13)
    if len(scores) < 2:
        txt = police.render("En attente de donnees...", True, GRIS)
        surface.blit(txt, (margin_x + 50, top_y + g_h // 2))
        return
        
    max_score = max(max(scores), 1)
    nb_pts = len(scores)
    
    pts_scores = []
    pts_moyennes = []
    
    for i in range(nb_pts):
        x = margin_x + int((i / (nb_pts - 1)) * g_w)
        y_s = base_y - int((scores[i] / max_score) * g_h)
        pts_scores.append((x, y_s))
        
        y_m = base_y - int((moyennes[i] / max_score) * g_h)
        pts_moyennes.append((x, y_m))
        
    if len(pts_scores) > 1:
        pygame.draw.lines(surface, (100, 100, 100), False, pts_scores, 1)
        pygame.draw.lines(surface, VERT_CLAIR, False, pts_moyennes, 2)
        
    lbl_max = police.render(f"Meilleur: {max_score}", True, ROUGE)
    surface.blit(lbl_max, (300, 10))
    lbl_avg = police.render(f"Moy(100): {moyennes[-1]:.1f}", True, VERT_CLAIR)
    surface.blit(lbl_avg, (300, 28))
    lbl_partie = police.render(f"Parties: {len(scores)}", True, BLANC)
    surface.blit(lbl_partie, (300, 46))


def main():
    pygame.init()
    fenetre_globale = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
    pygame.display.set_caption("🐍 Snake IA — Entraînement Double Vitesse (Calcul / Rendu)")
    horloge = pygame.time.Clock()
    
    surfaces = []
    for lig in range(NB_LIGNES):
        for col in range(NB_COLONNES):
            rect = pygame.Rect(col * LARGEUR_SOUS_JEU, lig * HAUTEUR_SOUS_JEU, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU)
            surfaces.append(fenetre_globale.subsurface(rect))
            
    offset_x = 3 * LARGEUR_SOUS_JEU
    offset_y = 1 * HAUTEUR_SOUS_JEU
    
    # Slider 1 : Vitesse d'entraînement (Arrière-plan - DQN steps par frame d'affichage)
    slider_train = Slider(30, 40, 240, 12, 1, 100, 10, titre="Calculs/Frame")
    
    # Slider 2 : Vitesse d'affichage (FPS de la démonstration en direct)
    slider_fps = Slider(30, 105, 240, 12, 5, 120, 30, titre="FPS Affichage")
    
    jeux = [SubSnakeGame(surfaces[i]) for i in range(7)]
    agent = AgentDQN()
    
    if os.path.exists(os.path.join("models", NOM_MODELE_DQN)):
        try:
            agent.charger()
        except Exception:
            pass

    scores_historique = []
    moyennes_historique = []
    liste_scores_recents = []
    
    en_pause = False
    en_cours = True
    
    etats = [jeu.get_state() for jeu in jeux]
    
    step_compteur = 0
    dernier_temps_fps = time.time()
    steps_depuis_dernier_temps = 0
    fps_reel = 0
    
    print("\nEntraînement double vitesse initialisé !")
    
    while en_cours:
        # --- Événements globaux et souris ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                en_cours = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    en_cours = False
                elif event.key == pygame.K_SPACE:
                    en_pause = not en_pause
                elif event.key == pygame.K_s:
                    agent.sauvegarder()
            
            # Mettre à jour les deux sliders
            slider_train.handle_event(event, offset_x, offset_y)
            slider_fps.handle_event(event, offset_x, offset_y)

        if en_pause:
            pygame.time.wait(100)
            continue
            
        # --- Découplage de la Vitesse de Calcul ---
        # Si le mode de calcul est au max, on boucle pendant un temps fixe (par exemple 16ms max par frame)
        # pour forcer le CPU à carburer à pleine puissance en tâche de fond.
        debut_frame_calcul = time.time()
        
        # Nombre de steps d'entraînement voulus par frame d'affichage
        nb_steps_a_faire = 1000 if slider_train.mode_max else int(slider_train.valeur)
        steps_faits = 0
        
        while steps_faits < nb_steps_a_faire:
            # Exécuter un cycle sur chaque serpent
            for i, jeu in enumerate(jeux):
                action = agent.choisir_action(etats[i], entrainement=True)
                etat_suivant, recompense, termine, score = jeu.step(action)
                agent.memoriser(etats[i], action, recompense, etat_suivant, termine)
                
                step_compteur += 1
                if step_compteur % 4 == 0:
                    agent.entrainer()
                    
                etats[i] = etat_suivant
                steps_depuis_dernier_temps += 1
                
                if termine:
                    scores_historique.append(score)
                    liste_scores_recents.append(score)
                    if len(liste_scores_recents) > 100:
                        liste_scores_recents.pop(0)
                    moyennes_historique.append(sum(liste_scores_recents) / len(liste_scores_recents))
                    
                    etats[i] = jeu.reset()
                    agent.fin_episode()
                    
                    if len(scores_historique) % 100 == 0:
                        agent.sauvegarder()
                        
            steps_faits += 1
            
            # En mode MAX, on s'arrête si on dépasse le budget temps d'une frame (ex: 15ms)
            # pour laisser l'affichage Pygame s'exécuter de façon fluide.
            if slider_train.mode_max and (time.time() - debut_frame_calcul > 0.015):
                break

        # --- Affichage Graphique synchrone ---
        # Rendre les 7 serpents
        for i, jeu in enumerate(jeux):
            jeu.render(id_jeu=i+1)
            
        # Mettre à jour les FPS réels de calcul toutes les secondes
        temps_actuel = time.time()
        if temps_actuel - dernier_temps_fps >= 1.0:
            fps_reel = int(steps_depuis_dernier_temps / (temps_actuel - dernier_temps_fps))
            steps_depuis_dernier_temps = 0
            dernier_temps_fps = temps_actuel
            
        # Dessiner le graphe et les deux curseurs dans la case 8
        dessiner_graphe_et_sliders(surfaces[7], scores_historique, moyennes_historique, slider_train, slider_fps)
        
        # Afficher la vitesse de calcul réelle dans le coin en haut à gauche
        txt_vitesse = pygame.font.SysFont('arial', 13).render(f"Calculs réels : {fps_reel} steps/s", True, VERT_CLAIR)
        surfaces[7].blit(txt_vitesse, (30, 140))
        
        # Mettre à jour l'écran global
        pygame.display.flip()
        
        # Rythmer l'affichage graphique selon le Slider 2 (FPS)
        horloge.tick(int(slider_fps.valeur))
            
    agent.sauvegarder()
    pygame.quit()
    print("Entraînement terminé proprement.")

if __name__ == "__main__":
    main()
