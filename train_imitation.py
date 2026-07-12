# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA MARL — Entraînement par Clonage de Comportement (Imitation)
=============================================================================
 Pré-entraîne de manière supervisée (Behavioral Cloning) les 4 architectures DQN
 (Standard, Profond, Élite, Collectif) sur le dataset généré par A* et Influence.
 Utilise la perte d'entropie croisée (Cross Entropy Loss) pour prédire l'action.
=============================================================================
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from marl_model import ReseauDQN

def train_imitation(epochs=15, batch_size=256, lr=0.001):
    print("=" * 70)
    print("     PRE-ENTRAINEMENT PAR IMITATION DES AGENTS DQN")
    print("=" * 70)
    
    device = torch.device("cpu")
    
    # Configuration des 5 agents a entraîner (4 en vision 25D, Raycast en 16D)
    modeles_config = {
        "Standard": {"couches_cachees": [256, 128], "taille_entree": 25, "chemin": "models/marl_dqn_standard.pth", "dataset": "data/imitation_dataset.pkl"},
        "Profond": {"couches_cachees": [256, 128, 64, 32], "taille_entree": 25, "chemin": "models/marl_dqn_profond.pth", "dataset": "data/imitation_dataset.pkl"},
        "Raycast": {"couches_cachees": [256, 128], "taille_entree": 16, "chemin": "models/marl_dqn_raycast.pth", "dataset": "data/raycast_dataset.pkl"},
        "Elite": {"couches_cachees": [256, 128, 64, 32], "taille_entree": 25, "chemin": "models/marl_dqn_elite.pth", "dataset": "data/imitation_dataset.pkl"},
        "Collectif": {"couches_cachees": [256, 128, 64, 32], "taille_entree": 25, "chemin": "models/marl_dqn_collectif.pth", "dataset": "data/imitation_dataset.pkl"}
    }
    
    # Boucle d'entraînement pour chaque réseau
    for nom, config in modeles_config.items():
        chemin_dataset = config["dataset"]
        if not os.path.exists(chemin_dataset):
            print(f"[!] Dataset introuvable pour [{nom}] : {chemin_dataset}")
            print(f"    Veuillez d'abord exécuter le générateur correspondant.")
            continue

        print(f"\n[Train] Agent [{nom}] (Architecture : {config['couches_cachees']}, entree {config['taille_entree']}D)")
        with open(chemin_dataset, "rb") as f:
            dataset = pickle.load(f)
        print(f"[Loader] {len(dataset)} transitions chargées depuis {chemin_dataset}.")

        np.random.shuffle(dataset)
        etats = np.array([x[0] for x in dataset], dtype=np.float32)
        actions = np.array([x[1] for x in dataset], dtype=np.int64)

        model = ReseauDQN(taille_entree=config["taille_entree"], couches_cachees=config["couches_cachees"], taille_sortie=3).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        num_samples = len(etats)
        
        for epoch in range(epochs):
            model.train()
            indices = np.random.permutation(num_samples)
            epoch_loss = 0.0
            correct = 0
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                x_batch = torch.tensor(etats[batch_indices], dtype=torch.float32, device=device)
                y_batch = torch.tensor(actions[batch_indices], dtype=torch.long, device=device)
                
                # Propagation avant
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                
                # Rétropropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                
            epoch_loss /= num_samples
            accuracy = correct / num_samples
            print(f"    Époque {epoch+1:02d}/{epochs:02d} | Loss : {epoch_loss:.4f} | Accuracy : {accuracy*100:.2f}%")
            
        # Sauvegarder les poids
        model.sauvegarder(config["chemin"])
        
        # Enregistrer les métadonnées de démarrage pour marl_arena.py
        chemin_meta = config["chemin"].replace(".pth", "_meta.pkl")
        with open(chemin_meta, "wb") as f:
            pickle.dump({
                'meilleure_moyenne': 6.5,  # Valeur de départ élevée grâce à l'imitation
                'nb_episodes': 200,        # Initialiser avec des épisodes
                'epsilon': 0.15,           # Commencer avec un epsilon faible (15% explo) pour exploiter les poids acquis
                'total_transitions': len(dataset)
            }, f)
        print(f"    [Saved] Poids et métadonnées sauvegardés pour [{nom}].")

    print("\n" + "=" * 70)
    print("[OK] Pré-entraînement par imitation terminé pour les 5 agents !")
    print("=" * 70)

if __name__ == "__main__":
    train_imitation()
