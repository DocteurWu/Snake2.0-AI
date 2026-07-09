# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA MARL — Définition des Réseaux de Neurones DQN
=============================================================================
 Modèle générique PyTorch pour s'adapter aux différentes architectures DQN.
=============================================================================
"""

import os
import torch
import torch.nn as nn

class ReseauDQN(nn.Module):
    """
    Réseau de neurones dense générique pour le Deep Q-Learning.
    """
    def __init__(self, taille_entree, couches_cachees, taille_sortie=3):
        """
        Initialise le réseau de neurones.
        
        Paramètres :
            taille_entree (int)      : Dimension de l'état (ex: 11 ou 24)
            couches_cachees (list)   : Liste des tailles de couches cachées (ex: [256, 128])
            taille_sortie (int)      : Nombre d'actions possibles (3)
        """
        super(ReseauDQN, self).__init__()
        
        tailles = [taille_entree] + couches_cachees + [taille_sortie]
        couches = []
        
        for i in range(len(tailles) - 1):
            couches.append(nn.Linear(tailles[i], tailles[i + 1]))
            # Applique ReLU uniquement sur les couches intermédiaires (pas sur la sortie)
            if i < len(tailles) - 2:
                couches.append(nn.ReLU())
                
        self.reseau = nn.Sequential(*couches)
        
    def forward(self, x):
        """Propagations avant (calcul des Q-values)."""
        return self.reseau(x)
        
    def sauvegarder(self, chemin_complet):
        """Sauvegarde les poids du modèle."""
        repertoire = os.path.dirname(chemin_complet)
        if repertoire:
            os.makedirs(repertoire, exist_ok=True)
        torch.save(self.state_dict(), chemin_complet)
        print(f"[OK] Modèle sauvegardé dans : {chemin_complet}")
        
    def charger(self, chemin_complet, device):
        """Charge les poids du modèle."""
        if os.path.exists(chemin_complet):
            self.load_state_dict(torch.load(chemin_complet, map_location=device))
            self.eval()
            print(f"[OK] Modèle chargé depuis : {chemin_complet}")
            return True
        else:
            print(f"[!] Fichier de modèle introuvable : {chemin_complet}")
            return False
