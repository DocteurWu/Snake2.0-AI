# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA MARL — Génération de Données de Démonstration (Imitation)
=============================================================================
 Ce script exécute le moteur de jeu en tâche de fond avec les agents experts
 Mathématicien (A*) et Économiste (Influence) pour collecter 200 000 transitions.
 Les états sont enregistrés au format Grille de Vision Locale 5x5 (25D).
=============================================================================
"""

import os
import pickle
import numpy as np
import random
from marl_game import MARLGame
from marl_agents import AStarAgent, EconomistAgent

def generate_data(target_samples_per_agent=100000):
    print("=" * 70)
    print("     GENERATION DU DATASET D'IMITATION (BEHAVIORAL CLONING)")
    print("=" * 70)
    
    # Initialiser le moteur de jeu (sans Pygame, exécution purement algorithmique)
    game = MARLGame(largeur=40, hauteur=40, nb_pommes=8)
    
    # Ajouter les agents experts
    game.ajouter_serpent(0, "Economiste", (0, 0, 0))
    game.ajouter_serpent(1, "Mathematicien", (0, 0, 0))
    game.initialiser_pommes()
    
    economist_agent = EconomistAgent()
    astar_agent = AStarAgent()
    
    dataset_eco = []
    dataset_math = []
    
    steps = 0
    while len(dataset_eco) < target_samples_per_agent or len(dataset_math) < target_samples_per_agent:
        # 1. Extraire les grilles de vision locale (25D) avant mouvement
        state_eco = game.get_vision_grid_state(0)
        state_math = game.get_vision_grid_state(1)
        
        # 2. Choisir l'action de chaque expert
        act_eco = economist_agent.choisir_action(game, 0)
        act_math = astar_agent.choisir_action(game, 1)
        
        # 3. Enregistrer les transitions (uniquement si le serpent est valide/vivant)
        if len(dataset_eco) < target_samples_per_agent and len(game.snakes[0]['corps']) > 0:
            dataset_eco.append((state_eco, act_eco))
            
        if len(dataset_math) < target_samples_per_agent and len(game.snakes[1]['corps']) > 0:
            dataset_math.append((state_math, act_math))
            
        # 4. Appliquer les actions et faire avancer la simulation
        actions = {0: act_eco, 1: act_math}
        game.step(actions)
        
        steps += 1
        if steps % 10000 == 0:
            print(f"[Dataset] Étape {steps} | Eco : {len(dataset_eco)}/{target_samples_per_agent} | Math : {len(dataset_math)}/{target_samples_per_agent}")
            
    # Combiner et mélanger le dataset final
    full_dataset = dataset_eco + dataset_math
    
    os.makedirs("data", exist_ok=True)
    chemin_dataset = "data/imitation_dataset.pkl"
    with open(chemin_dataset, "wb") as f:
        pickle.dump(full_dataset, f)
        
    print("=" * 70)
    print(f"[OK] Dataset généré avec succès dans : {chemin_dataset}")
    print(f"    Total : {len(full_dataset)} échantillons (50% Économiste, 50% Mathématicien)")
    print("=" * 70)

if __name__ == "__main__":
    generate_data()
