# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA 2.0 — Pipeline d'entraînement continu 24h (HEADLESS, sans Pygame)
=============================================================================
 Phase A : génération du dataset d'imitation (200k transitions, A* + Influence)
 Phase B : pré-entraînement par imitation (Behavioral Cloning) des 4 DQN
 Phase C : entraînement par renforcement continu (DQN) sur l'arène 8 agents
           pendant 24h, sauvegarde des checkpoints et du classement.
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

DUREE_SEC = 24 * 3600
# Heure de debut de Phase C. En mode RESUME, on recharge depuis le fichier
# pour que le compte a rebours 24h soit CONTINU malgre une coupure.
PHASE_C_START_FILE = "phase_c_start.txt"
if os.environ.get("RESUME") == "1" and os.path.exists(PHASE_C_START_FILE):
    with open(PHASE_C_START_FILE) as f:
        DEBUT = float(f.read().strip())
else:
    DEBUT = time.time()
LOG_PATH = "training_24h.log"

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

if os.environ.get("RESUME") == "1" and os.path.exists(PHASE_C_START_FILE):
    log(f"REPRISE : Phase C debut recharge depuis {PHASE_C_START_FILE} -> fin prevue a {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(DEBUT + DUREE_SEC))}")

# ---------------------------------------------------------------------------
# PHASE A : génération des datasets
# ---------------------------------------------------------------------------
if not os.path.exists("data/imitation_dataset.pkl"):
    log("PHASE A : génération du dataset d'imitation 25D (200k transitions)...")
    t0 = time.time()
    from generate_dataset import generate_data
    generate_data(target_samples_per_agent=100000)
    log(f"PHASE A : dataset 25D terminé en {time.time()-t0:.0f}s")
else:
    log("PHASE A : dataset 25D déjà présent, skip.")

if not os.path.exists("data/raycast_dataset.pkl"):
    log("PHASE A : génération du dataset d'imitation RAYCAST 16D (200k transitions)...")
    t0 = time.time()
    from generate_dataset_raycast import generate_data_raycast
    generate_data_raycast(target_samples_per_agent=100000)
    log(f"PHASE A : dataset raycast 16D terminé en {time.time()-t0:.0f}s")
else:
    log("PHASE A : dataset raycast déjà présent, skip.")

# ---------------------------------------------------------------------------
# PHASE B : pré-entraînement par imitation
# Skippée si RESUME=1 (les poids d'imitation sont déjà sur disque et
# Phase C les recharge via charger_si_existe()). Permet de reprendre un
# entraînement interrompu sans tout réécrire.
# ---------------------------------------------------------------------------
if os.environ.get("RESUME") == "1":
    log("PHASE B : SKIP (mode RESUME=1, les poids d'imitation sont déjà sur disque).")
else:
    log("PHASE B : pré-entraînement par imitation des 5 DQN...")
    t0 = time.time()
    from train_imitation import train_imitation
    train_imitation(epochs=20, batch_size=256, lr=0.001)
    log(f"PHASE B : terminée en {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# PHASE C : arène RL continue 24h (headless)
# ---------------------------------------------------------------------------
log("PHASE C : démarrage de l'arène RL continue 24h (headless)...")
# Sauvegarder l'heure de debut pour permettre une reprise continue (mode RESUME)
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

etape = 0
derniere_sauvegarde = DEBUT
derniere_classement = DEBUT

while time.time() - DEBUT < DUREE_SEC:
    etape += 1

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

    # Sauvegarde périodique (toutes les heures) + checkpoint horaire
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
                log(f"[!] Sauvegarde {sid} échouée : {e}")
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

# Sauvegarde finale
log("PHASE C : fin des 24h, sauvegarde finale...")
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
        log(f"[!] Sauvegarde finale {sid} échouée : {e}")

duree = time.time() - DEBUT
log(f"TERMINE en {duree/3600:.2f}h | {etape} ticks | modèles dans models/")
