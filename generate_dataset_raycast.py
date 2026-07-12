# -*- coding: utf-8 -*-
"""
Generation du dataset d'imitation 16D (RAYCAST) pour le serpent Raycast.
Memes experts (A* + Influence) que generate_dataset.py mais etats en
get_raycast_state (16D : 8 directions x [distance, id_objet]).
"""

import os
import pickle
import numpy as np
from marl_game import MARLGame
from marl_agents import AStarAgent, EconomistAgent

def generate_data_raycast(target_samples_per_agent=100000):
    print("=" * 70)
    print("  GENERATION DATASET RAYCAST 16D (IMITATION)")
    print("=" * 70)

    game = MARLGame(largeur=40, hauteur=40, nb_pommes=8)
    game.ajouter_serpent(0, "Economiste", (0, 0, 0))
    game.ajouter_serpent(1, "Mathematicien", (0, 0, 0))
    game.initialiser_pommes()

    economist_agent = EconomistAgent()
    astar_agent = AStarAgent()

    dataset_eco = []
    dataset_math = []
    steps = 0
    while len(dataset_eco) < target_samples_per_agent or len(dataset_math) < target_samples_per_agent:
        state_eco = game.get_raycast_state(0)
        state_math = game.get_raycast_state(1)
        act_eco = economist_agent.choisir_action(game, 0)
        act_math = astar_agent.choisir_action(game, 1)

        if len(dataset_eco) < target_samples_per_agent and len(game.snakes[0]['corps']) > 0:
            dataset_eco.append((state_eco, act_eco))
        if len(dataset_math) < target_samples_per_agent and len(game.snakes[1]['corps']) > 0:
            dataset_math.append((state_math, act_math))

        actions = {0: act_eco, 1: act_math}
        game.step(actions)
        steps += 1
        if steps % 10000 == 0:
            print(f"[Raycast-DS] Etape {steps} | Eco {len(dataset_eco)} | Math {len(dataset_math)}")

    full = dataset_eco + dataset_math
    os.makedirs("data", exist_ok=True)
    path = "data/raycast_dataset.pkl"
    with open(path, "wb") as f:
        pickle.dump(full, f)
    print(f"[OK] Dataset raycast 16D -> {path} ({len(full)} echantillons)")

if __name__ == "__main__":
    generate_data_raycast()
