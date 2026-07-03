# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Entraînement DQN découplé & Ultra-Haute Vitesse (Sans contention)
=============================================================================
 Version optimisée sans blocage de verrous (lock contention).
 Le verrou n'est acquis que pendant l'optimisation des poids du réseau.
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

# Verrou de modèle (utilisé uniquement pour protéger l'accès aux poids pendant train/inference)
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


# Fichiers et variables partagés
scores_train = []
moyennes_train = []
en_pause = False
en_cours = True
vitesse_reelle_bg = 0

def thread_entrainement_background(agent, slider_bg):
    global scores_train, moyennes_train, en_pause, en_cours, vitesse_reelle_bg
    
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

        # 1. Inférence Batchée (sécurisée par lock minimal)
        with lock_modele:
            actions = agent.choisir_actions_batch(etats, entrainement=True)

        # 2. Simulation physique des 20 jeux (HORS DU VERROU, ultra-rapide)
        for idx in range(nb_env):
            action = actions[idx]
            etat_actuel = etats[idx]
            
            # Pas de jeu logique
            etat_suivant, recompense, termine, score = jeux_invisibles[idx].step(action)
            
            # Stockage de la transition (HORS DU VERROU, thread-safe grâce au deque)
            agent.memoriser(etat_actuel, action, recompense, etat_suivant, termine)
            
            etats[idx] = etat_suivant
            steps_mesures += 1
            
            # 3. Entraînement DQN ponctuel (verrouillé uniquement pendant l'optimisation)
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
                
                etats[idx] = jeux_invisibles[idx].reset()
                with lock_modele:
                    agent.fin_episode()
                
                if len(scores_train) % 100 == 0:
                    with lock_modele:
                        agent.sauvegarder()
                        
        # Mesure des FPS réels de calcul en arrière-plan
        temps_actuel = time.time()
        if temps_actuel - dernier_temps >= 1.0:
            vitesse_reelle_bg = int(steps_mesures / (temps_actuel - dernier_temps))
            steps_mesures = 0
            dernier_temps = temps_actuel

        # Limitation de vitesse si demandée
        if fps_bg > 0:
            temps_ecoule = time.time() - temps_debut_step
            temps_attente = (1.0 / fps_bg) - temps_ecoule
            if temps_attente > 0:
                time.sleep(temps_attente)


def dessiner_graphe(surface, scores, moyennes, slider_bg, slider_vis):
    surface.fill((20, 20, 20))
    pygame.draw.rect(surface, BLEU, (0, 0, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU), 2)
    
    slider_bg.draw(surface)
    slider_vis.draw(surface)
    
    police = pygame.font.SysFont('arial', 13)
    
    # Bouton de pause / statut
    statut_txt = "STATUT : PAUSE" if en_pause else f"STATUT : RUNNING ({vitesse_reelle_bg} steps/s)"
    couleur_statut = ROUGE if en_pause else VERT_CLAIR
    surface.blit(police.render(statut_txt, True, couleur_statut), (30, 110))
    
    # Dessin du graphe d'évolution
    margin_x, margin_y = 50, 40
    g_w = LARGEUR_SOUS_JEU - margin_x - 20
    g_h = HAUTEUR_SOUS_JEU - margin_y - 170
    
    base_y = HAUTEUR_SOUS_JEU - margin_y
    top_y = 150
    
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (LARGEUR_SOUS_JEU - 20, base_y), 1)
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (margin_x, top_y), 1)
    
    if len(scores) < 2:
        txt = police.render("En attente de donnees du background...", True, GRIS)
        surface.blit(txt, (margin_x + 10, top_y + g_h // 2))
        return
        
    max_score = max(max(scores), 1)
    nb_pts = len(scores)
    
    pts_scores = []
    pts_moyennes = []
    
    pas = max(1, nb_pts // 500)
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
        pygame.draw.lines(surface, (100, 100, 100), False, pts_scores, 1)
        pygame.draw.lines(surface, VERT_CLAIR, False, pts_moyennes, 2)
        
    lbl_max = police.render(f"Meilleur: {max_score}", True, ROUGE)
    surface.blit(lbl_max, (300, 10))
    lbl_avg = police.render(f"Moy(100): {moyennes[-1]:.1f}", True, VERT_CLAIR)
    surface.blit(lbl_avg, (300, 28))
    lbl_partie = police.render(f"Parties BG: {len(scores)}", True, BLANC)
    surface.blit(lbl_partie, (300, 46))


def main():
    global en_pause, en_cours
    pygame.init()
    fenetre_globale = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
    pygame.display.set_caption("🐍 Snake IA — Double Contrôle de Vitesse Découplé (Entraînement / Rendu)")
    horloge = pygame.time.Clock()
    
    surfaces = []
    for lig in range(NB_LIGNES):
        for col in range(NB_COLONNES):
            rect = pygame.Rect(col * LARGEUR_SOUS_JEU, lig * HAUTEUR_SOUS_JEU, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU)
            surfaces.append(fenetre_globale.subsurface(rect))
            
    offset_x = 3 * LARGEUR_SOUS_JEU
    offset_y = 1 * HAUTEUR_SOUS_JEU
    
    slider_bg = Slider(30, 35, 240, 10, 10, 2000, 2000, titre="Train BG")
    slider_bg.mode_max = True # Débridé
    
    slider_vis = Slider(30, 80, 240, 10, 5, 120, 20, titre="Rendu Visuel")
    
    jeux_demo = [SubSnakeGame(surfaces[i]) for i in range(7)]
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
    
    print("\nSimulation démarrée (Inférence batchée ultra-haute vitesse sans verrous inutiles) !")
    
    while en_cours:
        fps_visual = int(slider_vis.valeur)

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

        # Rendu démo
        if not en_pause:
            for idx, jeu in enumerate(jeux_demo):
                with lock_modele:
                    action = agent.choisir_action(etats_demo[idx], entrainement=False)
                    
                etat_suivant, _, termine, _ = jeu.step(action)
                jeu.render(id_jeu=idx+1)
                etats_demo[idx] = etat_suivant
                
                if termine:
                    etats_demo[idx] = jeu.reset()

        dessiner_graphe(surfaces[7], scores_train, moyennes_train, slider_bg, slider_vis)
        
        pygame.display.flip()
        horloge.tick(fps_visual)
            
    en_cours = False
    thread_bg.join(timeout=1.0)
    with lock_modele:
        agent.sauvegarder()
    pygame.quit()
    print("Entraînement terminé.")

if __name__ == "__main__":
    main()
