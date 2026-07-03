# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Entraînement DQN avec Graphe Optimisé (Double Buffer) & Sélecteur
=============================================================================
 Interface légère avec sélecteur discret "[ ] Endless" pour basculer de mode.
 Rendu graphique optimisé sur image en cache pour éviter les chutes de framerate.
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
        police = pygame.font.SysFont('arial', 12)
        pygame.draw.rect(surface, GRIS, self.rect, border_radius=3)
        
        largeur_remplissage = self.bouton_rect.centerx - self.rect.x
        if largeur_remplissage > 0:
            remplissage_rect = pygame.Rect(self.rect.x, self.rect.y, largeur_remplissage, self.rect.height)
            pygame.draw.rect(surface, BLEU, remplissage_rect, border_radius=3)

        couleur_curseur = BLANC if not self.drag else VERT_CLAIR
        pygame.draw.rect(surface, couleur_curseur, self.bouton_rect, border_radius=2)
        
        val_txt = "MAX" if self.mode_max else f"{int(self.valeur)} FPS"
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
dernier_score_temps = 0
mode_plein_ecran_graphe = True
rafraichir_graphe = True # Flag pour recalculer l'image du graphe uniquement si nécessaire

def thread_entrainement_background(agent, slider_bg):
    global scores_train, moyennes_train, en_pause, en_cours, vitesse_reelle_bg, dernier_score_temps, rafraichir_graphe
    
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
                
                dernier_score_temps = time.time()
                rafraichir_graphe = True # Demander la reconstruction du graphe au thread principal
                
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


# Buffers d'image en mémoire cache pour le graphe
cache_surface_graphe = None

def generer_rendu_graphe(large, haut, scores, moyennes):
    """Dessine le graphe brut sur une surface hors-écran dédiée (mise en cache)."""
    g_surf = pygame.Surface((large, haut))
    g_surf.fill((20, 20, 20))
    pygame.draw.rect(g_surf, BLEU, (0, 0, large, haut), 3)
    
    police = pygame.font.SysFont('arial', 13)
    
    margin_x = 60
    margin_y = 50
    g_w = large - margin_x - 40
    g_h = haut - margin_y - 170
    
    base_y = haut - margin_y
    top_y = 150
    
    # Axes
    pygame.draw.line(g_surf, BLANC, (margin_x, base_y), (large - 40, base_y), 2)
    pygame.draw.line(g_surf, BLANC, (margin_x, base_y), (margin_x, top_y), 2)
    
    if len(scores) < 2:
        txt = police.render("En attente de donnees du background...", True, GRIS)
        g_surf.blit(txt, (large // 2 - 100, top_y + g_h // 2))
        return g_surf
        
    max_score = max(max(scores), 1)
    nb_pts = len(scores)
    
    pts_scores = []
    pts_moyennes = []
    
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
        
    if len(pts_scores) > 1:
        pygame.draw.lines(g_surf, (80, 80, 80), False, pts_scores, 1)
        pygame.draw.lines(g_surf, VERT_CLAIR, False, pts_moyennes, 2)
        
    # Infos
    lbl_max = police.render(f"Meilleur score : {max_score}", True, ROUGE)
    lbl_avg = police.render(f"Moyenne (100) : {moyennes[-1]:.1f}", True, VERT_CLAIR)
    lbl_partie = police.render(f"Episodes : {len(scores)}", True, BLANC)
    
    g_surf.blit(lbl_max, (large - 180, 15))
    g_surf.blit(lbl_avg, (large - 180, 32))
    g_surf.blit(lbl_partie, (large - 180, 49))
    
    return g_surf


def dessiner_interface_graphe(surface, slider_bg, slider_vis, large, haut):
    """Affiche le graphe en cache et ajoute la surbrillance dynamique du dernier point."""
    global cache_surface_graphe, rafraichir_graphe
    
    # 1. Régénérer le cache du graphe si nécessaire (mort de serpent ou changement de taille)
    if cache_surface_graphe is None or rafraichir_graphe or cache_surface_graphe.get_size() != (large, haut):
        cache_surface_graphe = generer_rendu_graphe(large, haut, scores_train, moyennes_train)
        rafraichir_graphe = False
        
    # Coller le fond du graphe mis en cache
    surface.blit(cache_surface_graphe, (0, 0))
    
    # 2. Dessiner les éléments interactifs par-dessus (pas mis en cache car ils bougent)
    slider_bg.draw(surface)
    slider_vis.draw(surface)
    
    # Dessiner la checkbox Endless discrète
    police = pygame.font.SysFont('arial', 13, bold=True)
    # Fond de la checkbox
    checkbox_rect = pygame.Rect(30, 15, 16, 16)
    pygame.draw.rect(surface, GRIS, checkbox_rect, border_radius=3)
    if mode_plein_ecran_graphe:
        # Remplir en bleu/vert si coché
        pygame.draw.rect(surface, VERT_CLAIR, (34, 19, 8, 8), border_radius=1)
    pygame.draw.rect(surface, BLANC, checkbox_rect, 1, border_radius=3)
    
    surface.blit(police.render("Endless", True, BLANC), (54, 15))
    
    # Statut
    police_statut = pygame.font.SysFont('arial', 12)
    statut_txt = "STATUT : PAUSE [ESPACE]" if en_pause else f"VITESSE : {vitesse_reelle_bg} steps/s"
    couleur_statut = ROUGE if en_pause else VERT_CLAIR
    surface.blit(police_statut.render(statut_txt, True, couleur_statut), (130, 16))

    # 3. Dessiner l'effet de flash sur le dernier point (si le graphe contient des points)
    if len(scores_train) > 0:
        margin_x = 60
        margin_y = 50
        g_w = large - margin_x - 40
        g_h = haut - margin_y - 170
        base_y = haut - margin_y
        
        max_score = max(max(scores_train), 1)
        
        # Position du dernier point
        dernier_pt_x = margin_x + g_w
        dernier_pt_y = base_y - int((scores_train[-1] / max_score) * g_h)
        
        # Animation
        t_ecoule = time.time() - dernier_score_temps
        if t_ecoule < 0.25:
            rayon_flash = int(14 * (1.0 - t_ecoule / 0.25))
            surface_halo = pygame.Surface((rayon_flash*2, rayon_flash*2), pygame.SRCALPHA)
            pygame.draw.circle(surface_halo, (0, 200, 255, 130), (rayon_flash, rayon_flash), rayon_flash)
            surface.blit(surface_halo, (dernier_pt_x - rayon_flash, dernier_pt_y - rayon_flash))
        
        pygame.draw.circle(surface, BLANC, (dernier_pt_x, dernier_pt_y), 5)
        pygame.draw.circle(surface, ROUGE, (dernier_pt_x, dernier_pt_y), 3)


def main():
    global en_pause, en_cours, mode_plein_ecran_graphe, rafraichir_graphe
    pygame.init()
    fenetre_globale = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
    pygame.display.set_caption("🐍 Snake IA — Interface Graphique double-buffer optimisée")
    horloge = pygame.time.Clock()
    
    surfaces_grille = []
    for lig in range(NB_LIGNES):
        for col in range(NB_COLONNES):
            rect = pygame.Rect(col * LARGEUR_SOUS_JEU, lig * HAUTEUR_SOUS_JEU, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU)
            surfaces_grille.append(fenetre_globale.subsurface(rect))
            
    rect_plein_ecran = pygame.Rect(0, 0, LARGEUR_FENETRE, HAUTEUR_FENETRE)
    surface_plein_ecran = fenetre_globale.subsurface(rect_plein_ecran)
    
    # Sliders repositionnés proprement
    slider_bg = Slider(30, 50, 240, 8, 10, 2000, 2000, titre="Train BG")
    slider_bg.mode_max = True
    
    slider_vis = Slider(30, 90, 240, 8, 5, 120, 20, titre="Rendu Visuel")
    
    jeux_demo = [SubSnakeGame(surfaces_grille[i]) for i in range(7)]
    etats_demo = [jeu.reset() for jeu in jeux_demo]
    
    agent = AgentDQN()
    if os.path.exists(os.path.join("models", NOM_MODELE_DQN)):
        try:
            agent.charger()
        except Exception:
            pass

    thread_bg = threading.Thread(
        target=thread_entrainement_background, 
        args=(agent, slider_bg),
        daemon=True
    )
    thread_bg.start()
    
    print("\nSimulation démarrée (Rendu double-buffer optimisé) !")
    
    while en_cours:
        fps_visual = int(slider_vis.valeur)
        
        # Positionnement des offsets de clic
        if mode_plein_ecran_graphe:
            offset_x, offset_y = 0, 0
        else:
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
            
            slider_bg.handle_event(event, offset_x, offset_y)
            slider_vis.handle_event(event, offset_x, offset_y)
            
            # Gestion du clic sur la checkbox "Endless"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = (event.pos[0] - offset_x, event.pos[1] - offset_y)
                # Zone de la Checkbox (30, 15) à (120, 31) incluant le texte
                if 30 <= pos[0] <= 120 and 12 <= pos[1] <= 32:
                    mode_plein_ecran_graphe = not mode_plein_ecran_graphe
                    rafraichir_graphe = True
                    fenetre_globale.fill(NOIR)

        # 1. Rendu Démo (uniquement si NON-Endless)
        if not mode_plein_ecran_graphe and not en_pause:
            for idx, jeu in enumerate(jeux_demo):
                with lock_modele:
                    action = agent.choisir_action(etats_demo[idx], entrainement=False)
                etat_suivant, _, termine, _ = jeu.step(action)
                jeu.render(id_jeu=idx+1)
                etats_demo[idx] = etat_suivant
                if termine:
                    etats_demo[idx] = jeu.reset()

        # 2. Rendu Interface Graphe & Sliders
        if mode_plein_ecran_graphe:
            dessiner_interface_graphe(surface_plein_ecran, slider_bg, slider_vis, LARGEUR_FENETRE, HAUTEUR_FENETRE)
        else:
            dessiner_interface_graphe(surfaces_grille[7], slider_bg, slider_vis, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU)
            
        pygame.display.flip()
        
        # Le taux de rafraîchissement visuel du thread principal est régulé à 30 FPS en mode Graphe Plein écran.
        # Cela évite de consommer le CPU en appels de rafraîchissement d'affichage inutiles, tout en gardant 
        # le thread de background à pleine puissance en tâche de fond.
        if mode_plein_ecran_graphe:
            horloge.tick(30)
        else:
            horloge.tick(fps_visual)
            
    en_cours = False
    thread_bg.join(timeout=1.0)
    with lock_modele:
        agent.sauvegarder()
    pygame.quit()
    print("Entraînement terminé.")

if __name__ == "__main__":
    main()
