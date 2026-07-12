# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA 2.0 — TEST CURRICULUM (option A + B)
=============================================================================
 BUT : tester deux changements d'approche sans casser le pipeline 24h.
   A) Reward shaping enrichi   -> deja dans marl_game.py (step)
   B) Curriculum learning       -> ici : DQN-vs-DQN d'abord, puis mix experts

 DUREE test = 3h (CURRICULUM_TICKS puis arrene mixte jusqu'a la fin).
 Reprend les poids DQN existants (charger_si_existe) comme point de depart.
=============================================================================
"""

import os
import sys
import time
import pickle
import random
import numpy as np
import torch

from marl_game import MARLGame, DROITE, BAS, GAUCHE, HAUT
from marl_agents import DQNAgent, AStarAgent, EconomistAgent, StrategistAgent

# --- Parametres test ---
DUREE_SEC = 3 * 3600
CURRICULUM_TICKS = 4000   # ~75 min de DQN-vs-DQN seul avant d'introduire les experts
PHASE_C_START_FILE = "phase_c_start.txt"

if os.environ.get("RESUME") == "1" and os.path.exists(PHASE_C_START_FILE):
    with open(PHASE_C_START_FILE) as f:
        DEBUT = float(f.read().strip())
else:
    DEBUT = time.time()
LOG_PATH = "training_curriculum.log"

COULEURS = [
    (80, 180, 255),   # 0 Standard
    (255, 60, 60),    # 1 Profond
    (200, 60, 200),   # 2 Raycast
    (50, 240, 50),    # 3 Mathématicien
    (240, 220, 40),   # 4 Élite
    (255, 130, 0),    # 5 Économiste
    (40, 220, 220),   # 6 Stratège
    (110, 110, 255),  # 7 Collectif
]

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

# --- Phase C ---
log("CURRICULUM : demarrage (A=reward shaping actif, B=curriculum)...")
try:
    with open(PHASE_C_START_FILE, "w") as f:
        f.write(str(DEBUT))
except Exception:
    pass

jeu = MARLGame(40, 40, nb_pommes=8)
jeu.ajouter_serpent(0, "Standard (DQN)", COULEURS[0])
jeu.ajouter_serpent(1, "Profond (DQN)", COULEURS[1])
jeu.ajouter_serpent(2, "Raycast (DQN)", COULEURS[2])
jeu.ajouter_serpent(3, "Mathématicien (A*)", COULEURS[3])
jeu.ajouter_serpent(4, "Élite (DQN Sélectif)", COULEURS[4])
jeu.ajouter_serpent(5, "Économiste (Influence)", COULEURS[5])
jeu.ajouter_serpent(6, "Stratège (GameTheory)", COULEURS[6])
jeu.ajouter_serpent(7, "Collectif (DQN Partagé)", COULEURS[7])
jeu.initialiser_pommes()

agents = {
    0: DQNAgent("Standard", taille_entree=25, couches_cachees=[256, 128]),
    1: DQNAgent("Profond", taille_entree=25, couches_cachees=[256, 128, 64, 32], epsilon_decay=0.9995),
    2: DQNAgent("Raycast", taille_entree=16, couches_cachees=[256, 128]),
    3: AStarAgent(),
    4: DQNAgent("Elite", taille_entree=25, couches_cachees=[256, 128, 64, 32], epsilon_decay=0.9995),
    5: EconomistAgent(),
    6: StrategistAgent(),
    7: DQNAgent("Collectif", taille_entree=25, couches_cachees=[256, 128, 64, 32], epsilon_decay=0.9995),
}

for sid in [0, 1, 2, 4, 7]:
    agents[sid].charger_si_existe()

# --- Curriculum B : desactiver les experts au depart ---
EXPERTS = [3, 5, 6]
for sid in EXPERTS:
    jeu.snakes[sid]['actif'] = False
phase_curriculum_finie = False

etape = 0
derniere_sauvegarde = DEBUT
derniere_classement = DEBUT

while time.time() - DEBUT < DUREE_SEC:
    etape += 1

    # Bascule curriculum -> mix apres CURRICULUM_TICKS
    if not phase_curriculum_finie and etape >= CURRICULUM_TICKS:
        for sid in EXPERTS:
            jeu.snakes[sid]['actif'] = True
            # respawn propre si mort
            if not jeu.snakes[sid].get('corps'):
                jeu.respawn_serpent(sid)
        phase_curriculum_finie = True
        log(f"CURRICULUM : fin phase DQN-vs-DQN @ tick {etape} -> experts ACTIVES (mix)")

    etats_25d_tous = {}
    for sid in range(8):
        if jeu.snakes[sid].get('actif', True):
            etats_25d_tous[sid] = jeu.get_vision_grid_state(sid)

    etats_courants = {}
    for sid, s in jeu.snakes.items():
        if not s.get('actif', True):
            continue
        if sid in [0, 1, 4, 7]:
            etats_courants[sid] = etats_25d_tous[sid]
        elif sid == 2:
            etats_courants[sid] = jeu.get_raycast_state(sid)

    actions = {}
    for sid in range(8):
        if not jeu.snakes[sid].get('actif', True):
            continue
        if sid in [0, 1, 4, 7]:
            actions[sid] = agents[sid].choisir_action(etats_courants[sid])
        elif sid == 2:
            actions[sid] = agents[sid].choisir_action(etats_courants[sid])
        else:
            actions[sid] = agents[sid].choisir_action(jeu, sid)

    morts, recompenses = jeu.step(actions)

    next_states_25d = {}
    for sid in range(8):
        if jeu.snakes[sid].get('actif', True):
            next_states_25d[sid] = jeu.get_vision_grid_state(sid)
    next_state_ray = None
    if jeu.snakes[2].get('actif', True):
        next_state_ray = jeu.get_raycast_state(2)

    # Standard
    if jeu.snakes[0].get('actif', True):
        done1 = (0 in morts)
        agents[0].memoriser(etats_courants[0], actions[0], recompenses[0], next_states_25d[0], done1)
        for _ in range(8):
            agents[0].entrainer()
        if done1:
            agents[0].fin_episode()
            agents[0].sauvegarder_si_meilleur(jeu.obtenir_moyenne_10_vies(0))

    # Profond
    if jeu.snakes[1].get('actif', True):
        done2 = (1 in morts)
        agents[1].memoriser(etats_courants[1], actions[1], recompenses[1], next_states_25d[1], done2)
        for _ in range(8):
            agents[1].entrainer()
        if done2:
            agents[1].fin_episode()
            agents[1].sauvegarder_si_meilleur(jeu.obtenir_moyenne_10_vies(1))

    # Raycast
    if jeu.snakes[2].get('actif', True):
        done3 = (2 in morts)
        agents[2].memoriser(etats_courants[2], actions[2], recompenses[2], next_state_ray, done3)
        for _ in range(8):
            agents[2].entrainer()
        if done3:
            agents[2].fin_episode()
            agents[2].sauvegarder_si_meilleur(jeu.obtenir_moyenne_10_vies(2))

    # Meilleur serpent actuel
    meilleur_sid = None
    meilleure_moy = -1.0
    for sid, s in jeu.snakes.items():
        if s.get('actif', True):
            moy_s = jeu.obtenir_moyenne_10_vies(sid)
            if moy_s > meilleure_moy:
                meilleure_moy = moy_s
                meilleur_sid = sid

    # Élite
    if jeu.snakes[4].get('actif', True):
        done4 = (4 in morts)
        agents[4].memoriser(etats_courants[4], actions[4], recompenses[4], next_states_25d[4], done4)
        if meilleur_sid is not None and meilleur_sid != 4:
            done_best = (meilleur_sid in morts)
            agents[4].memoriser(etats_25d_tous[meilleur_sid], actions[meilleur_sid], recompenses[meilleur_sid], next_states_25d[meilleur_sid], done_best)
        for _ in range(8):
            agents[4].entrainer()
        if done4:
            agents[4].fin_episode()
            agents[4].sauvegarder_si_meilleur(jeu.obtenir_moyenne_10_vies(4))

    # Collectif
    if jeu.snakes[7].get('actif', True):
        for sid in range(8):
            if jeu.snakes[sid].get('actif', True):
                done_sid = (sid in morts)
                agents[7].memoriser(etats_25d_tous[sid], actions[sid], recompenses[sid], next_states_25d[sid], done_sid)
        for _ in range(8):
            agents[7].entrainer()
        done7 = (7 in morts)
        if done7:
            agents[7].fin_episode()
            agents[7].sauvegarder_si_meilleur(jeu.obtenir_moyenne_10_vies(7))

    # Sauvegarde périodique (toutes les heures)
    if time.time() - derniere_sauvegarde >= 3600:
        derniere_sauvegarde = time.time()
        for sid in [0, 1, 2, 4, 7]:
            try:
                agents[sid].reseau_principal.sauvegarder(agents[sid].chemin_sauvegarde)
                mp = agents[sid].chemin_sauvegarde.replace(".pth", "_meta.pkl")
                with open(mp, "wb") as f:
                    pickle.dump({'meilleure_moyenne': agents[sid].meilleure_moyenne,
                                 'nb_episodes': agents[sid].nb_episodes,
                                 'epsilon': agents[sid].epsilon,
                                 'total_transitions': agents[sid].total_transitions}, f)
            except Exception as e:
                log(f"[!] Sauvegarde {sid} echouee : {e}")
        log(f"[SAVE] Checkpoint horaire @ tick {etape}")

    # Classement périodique (toutes les 10 min)
    if time.time() - derniere_classement >= 600:
        derniere_classement = time.time()
        stats = []
        for sid, s in jeu.snakes.items():
            eps = agents[sid].epsilon if sid in [0, 1, 2, 4, 7] else None
            stats.append((sid, s['nom'], jeu.obtenir_moyenne_10_vies(sid), s['max_longueur'], s['nb_respawns'], eps))
        stats.sort(key=lambda x: (x[2], x[3]), reverse=True)
        lignes = " | ".join(f"{n.split(' ')[0]}={moy:.2f}(rec {rec})" for _, n, moy, _, rec, _ in stats)
        log(f"[RANK] tick {etape} | {lignes}")
        log(f"[EPS] " + " | ".join(f"{n.split(' ')[0]}:ε={eps:.3f}" for _, n, _, _, _, eps in stats if eps is not None))
        if not phase_curriculum_finie:
            log(f"[CURRICULUM] DQN-vs-DQN en cours ({etape}/{CURRICULUM_TICKS})")

log("CURRICULUM : fin du test, sauvegarde finale...")
for sid in [0, 1, 2, 4, 7]:
    try:
        agents[sid].reseau_principal.sauvegarder(agents[sid].chemin_sauvegarde)
        mp = agents[sid].chemin_sauvegarde.replace(".pth", "_meta.pkl")
        with open(mp, "wb") as f:
            pickle.dump({'meilleure_moyenne': agents[sid].meilleure_moyenne,
                         'nb_episodes': agents[sid].nb_episodes,
                         'epsilon': agents[sid].epsilon,
                         'total_transitions': agents[sid].total_transitions}, f)
    except Exception as e:
        log(f"[!] Sauvegarde finale {sid} echouee : {e}")

duree = time.time() - DEBUT
log(f"TERMINE en {duree/3600:.2f}h | {etape} ticks | modele dans models/")
