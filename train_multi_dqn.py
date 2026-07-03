# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Entraînement DQN Multi-Agent Parallèle (Optimisé Vitesse)
=============================================================================
 Fait tourner 7 serpents en parallèle.
 Si Vitesse = MAX : désactive le rendu du jeu pour un calcul CPU pur à 
 vitesse maximale (head-less dynamique), ce qui évite le blocage VSync.
 
 Contrôles :
   Clic souris : Ajuster la vitesse avec le Slider dans la 8ème case
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
    def __init__(self, x, y, largeur, hauteur, val_min, val_max, val_init):
        self.rect = pygame.Rect(x, y, largeur, hauteur)
        self.min = val_min
        self.max = val_max
        self.valeur = val_init
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
        
        val_txt = "MAX (No limit - Headless)" if self.mode_max else f"{int(self.valeur)} FPS"
        txt = police.render(f"Vitesse : {val_txt}", True, BLANC)
        surface.blit(txt, (self.rect.x, self.rect.y - 20))

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


def dessiner_graphe(surface, scores, moyennes, slider):
    surface.fill((20, 20, 20))
    pygame.draw.rect(surface, BLEU, (0, 0, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU), 2)
    
    slider.draw(surface)
    
    police = pygame.font.SysFont('arial', 13)
    b_50 = police.render("[50 FPS]", True, VERT_CLAIR)
    b_max = police.render("[Vitesse MAX]", True, ROUGE)
    surface.blit(b_50, (30, 75))
    surface.blit(b_max, (140, 75))
    
    margin_x, margin_y = 50, 40
    g_w = LARGEUR_SOUS_JEU - margin_x - 20
    g_h = HAUTEUR_SOUS_JEU - margin_y - 140
    
    base_y = HAUTEUR_SOUS_JEU - margin_y
    top_y = 130
    
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (LARGEUR_SOUS_JEU - 20, base_y), 1)
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (margin_x, top_y), 1)
    
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
    pygame.display.set_caption("🐍 Snake IA — Entraînement RL Multi-Agent (Optimisation Vitesse)")
    horloge = pygame.time.Clock()
    
    surfaces = []
    for lig in range(NB_LIGNES):
        for col in range(NB_COLONNES):
            rect = pygame.Rect(col * LARGEUR_SOUS_JEU, lig * HAUTEUR_SOUS_JEU, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU)
            surfaces.append(fenetre_globale.subsurface(rect))
            
    offset_x = 3 * LARGEUR_SOUS_JEU
    offset_y = 1 * HAUTEUR_SOUS_JEU
    
    slider = Slider(30, 40, 240, 12, 5, 1000, 60)
    
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
    
    print("\nEntraînement multi-agents optimisé lancé !")
    
    while en_cours:
        fps_ia = 0 if slider.mode_max else int(slider.valeur)

        # --- Gestion événements ---
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
            
            slider.handle_event(event, offset_x, offset_y)
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = (event.pos[0] - offset_x, event.pos[1] - offset_y)
                if 30 <= pos[0] <= 110 and 70 <= pos[1] <= 95:
                    slider.valeur = 50
                    slider.mode_max = False
                    slider.update_bouton_pos()
                elif 140 <= pos[0] <= 240 and 70 <= pos[1] <= 95:
                    slider.mode_max = True
                    slider.valeur = slider.max
                    slider.update_bouton_pos()

        if en_pause:
            pygame.time.wait(100)
            continue
            
        # --- Boucle physique / calculs ---
        for i, jeu in enumerate(jeux):
            action = agent.choisir_action(etats[i], entrainement=True)
            etat_suivant, recompense, termine, score = jeu.step(action)
            agent.memoriser(etats[i], action, recompense, etat_suivant, termine)
            
            # OPTIMISATION 1 : On entraîne le réseau 1 fois toutes les 4 étapes physiques
            # de façon distribuée pour soulager grandement le CPU sans ralentir la convergence.
            step_compteur += 1
            if step_compteur % 4 == 0:
                agent.entrainer()
            
            # OPTIMISATION 2 : Si mode_max, on désactive complètement les appels de rendu 
            # individuels Pygame des cases de jeu (headless dynamique).
            if not slider.mode_max:
                jeu.render(id_jeu=i+1)
                
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

        # Calculer le framerate de calcul réel toutes les secondes
        temps_actuel = time.time()
        if temps_actuel - dernier_temps_fps >= 1.0:
            fps_reel = int(steps_depuis_dernier_temps / (temps_actuel - dernier_temps_fps))
            steps_depuis_dernier_temps = 0
            dernier_temps_fps = temps_actuel

        # Rendu de la 8ème case (graphe et slider)
        if slider.mode_max:
            # En mode MAX, on nettoie tout l'écran avec un message de veille pour éviter de forcer sur le GPU/VSync
            fenetre_globale.fill(NOIR)
            
            # Rendre uniquement la case #8 et un message informatif au milieu de l'écran
            police_grand = pygame.font.SysFont('arial', 24)
            info_txt = police_grand.render(f"Calculs GPU/CPU sans VSync -- Vitesse reelle : {fps_reel} FPS (steps/sec)", True, VERT_CLAIR)
            txt_help = pygame.font.SysFont('arial', 16).render("Pour ré-afficher les serpents en direct, déplacez le slider vers la gauche ou cliquez sur [50 FPS].", True, BLANC)
            
            fenetre_globale.blit(info_txt, (50, HAUTEUR_FENETRE // 2 - 40))
            fenetre_globale.blit(txt_help, (50, HAUTEUR_FENETRE // 2))
            
            # Toujours dessiner le graphe et le slider dans la case 8 pour pouvoir interagir
            dessiner_graphe(surfaces[7], scores_historique, moyennes_historique, slider)
        else:
            # Rendu classique du graphe
            dessiner_graphe(surfaces[7], scores_historique, moyennes_historique, slider)
        
        # Mettre à jour l'affichage global
        pygame.display.flip()
        
        # Rythme d'affichage
        if fps_ia > 0:
            horloge.tick(fps_ia)
            
    agent.sauvegarder()
    pygame.quit()
    print("Entraînement terminé proprement.")

if __name__ == "__main__":
    main()
