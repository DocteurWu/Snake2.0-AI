# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Agent DQN (Deep Q-Network)
=============================================================================
 Agent d'apprentissage par renforcement avec :
 - Replay Buffer (mémoire d'expériences)
 - Politique epsilon-greedy (exploration/exploitation)
 - Réseau cible (stabilité de l'entraînement)
 Optimisé pour CPU ARM (Orange Pi 3B).
=============================================================================
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    TAILLE_ETAT, NB_ACTIONS, DEVICE,
    TAILLE_MEMOIRE, TAILLE_BATCH, GAMMA,
    EPSILON_DEBUT, EPSILON_FIN, EPSILON_DECROISSANCE,
    TAUX_APPRENTISSAGE, MISE_A_JOUR_CIBLE,
    NOM_MODELE_DQN
)
from model import ReseauSerpent


class MemoireReplay:
    """
    Buffer circulaire pour stocker les transitions (s, a, r, s', done).
    Utilise un deque pour une gestion mémoire efficace.
    """
    
    def __init__(self, capacite=TAILLE_MEMOIRE):
        """Initialise le buffer avec une capacité maximale."""
        self.memoire = deque(maxlen=capacite)
    
    def ajouter(self, etat, action, recompense, etat_suivant, termine):
        """Ajoute une transition dans la mémoire."""
        self.memoire.append((etat, action, recompense, etat_suivant, termine))
    
    def echantillonner(self, taille_batch):
        """
        Échantillonne un mini-batch aléatoire de transitions.
        
        Retourne :
            Tuple de Tensors : (etats, actions, recompenses, etats_suivants, termines)
        """
        batch = random.sample(self.memoire, taille_batch)
        
        etats, actions, recompenses, etats_suivants, termines = zip(*batch)
        
        return (
            torch.tensor(np.array(etats), dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.long, device=DEVICE),
            torch.tensor(recompenses, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(etats_suivants), dtype=torch.float32, device=DEVICE),
            torch.tensor(termines, dtype=torch.bool, device=DEVICE)
        )
    
    def __len__(self):
        return len(self.memoire)


class AgentDQN:
    """
    Agent Deep Q-Network complet.
    
    Fonctionnalités :
        - Double réseau (principal + cible) pour la stabilité
        - Politique epsilon-greedy avec décroissance
        - Replay buffer pour casser les corrélations temporelles
        - Entraînement par mini-batches
    """
    
    def __init__(self):
        # --- Réseaux de neurones ---
        self.reseau_principal = ReseauSerpent()   # Réseau qui apprend
        self.reseau_cible = ReseauSerpent()       # Réseau cible (stabilité)
        self._synchroniser_cible()                 # Copie initiale
        
        # --- Optimiseur (Adam : efficace et adaptatif) ---
        self.optimiseur = optim.Adam(
            self.reseau_principal.parameters(),
            lr=TAUX_APPRENTISSAGE
        )
        
        # --- Fonction de perte ---
        self.critere = nn.MSELoss()
        
        # --- Mémoire de replay ---
        self.memoire = MemoireReplay()
        
        # --- Exploration (epsilon-greedy) ---
        self.epsilon = EPSILON_DEBUT
        
        # --- Compteurs ---
        self.nb_episodes = 0
        self.nb_steps_entrainement = 0
    
    def _synchroniser_cible(self):
        """Copie les poids du réseau principal vers le réseau cible."""
        self.reseau_cible.load_state_dict(self.reseau_principal.state_dict())
        self.reseau_cible.eval()  # Le réseau cible ne s'entraîne jamais
    
    def choisir_action(self, etat, entrainement=True):
        """
        Choisit une action individuelle selon la politique epsilon-greedy.
        """
        if entrainement and random.random() < self.epsilon:
            return random.randint(0, NB_ACTIONS - 1)
        
        with torch.no_grad():
            etat_tensor = torch.tensor(
                etat, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            q_values = self.reseau_principal(etat_tensor)
            return q_values.argmax(dim=1).item()

    def choisir_actions_batch(self, etats, entrainement=True):
        """
        Choisit des actions pour un lot (batch) d'états d'un seul coup.
        Optimisation critique pour maximiser la vitesse d'inférence sur CPU.
        
        Paramètres :
            etats (list ou np.array) : Liste des N états de taille [N, 11]
            entrainement (bool)      : Si True, applique epsilon-greedy individuellement
            
        Retourne :
            np.array : Liste des N actions choisies de taille [N]
        """
        nb_etats = len(etats)
        actions = np.zeros(nb_etats, dtype=np.int64)
        
        # Identifier les indices qui exploiteront vs ceux qui exploreront
        indices_exploitation = []
        for idx in range(nb_etats):
            if entrainement and random.random() < self.epsilon:
                actions[idx] = random.randint(0, NB_ACTIONS - 1)
            else:
                indices_exploitation.append(idx)
                
        # Exécuter l'inférence groupée pour toutes les exploitations
        if len(indices_exploitation) > 0:
            etats_a_predire = np.array([etats[i] for i in indices_exploitation], dtype=np.float32)
            with torch.no_grad():
                etats_tensor = torch.tensor(etats_a_predire, dtype=torch.float32, device=DEVICE)
                q_values = self.reseau_principal(etats_tensor)
                predictions = q_values.argmax(dim=1).cpu().numpy()
                
            for i, idx in enumerate(indices_exploitation):
                actions[idx] = predictions[i]
                
        return actions
    
    def memoriser(self, etat, action, recompense, etat_suivant, termine):
        """Stocke une transition dans le replay buffer."""
        self.memoire.ajouter(etat, action, recompense, etat_suivant, termine)
    
    def entrainer(self):
        """
        Entraîne le réseau principal sur un mini-batch du replay buffer.
        """
        if len(self.memoire) < TAILLE_BATCH:
            return None
        
        etats, actions, recompenses, etats_suivants, termines = (
            self.memoire.echantillonner(TAILLE_BATCH)
        )
        
        q_actuelles = self.reseau_principal(etats)
        q_actuelles = q_actuelles.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            q_suivantes = self.reseau_cible(etats_suivants)
            q_max_suivantes = q_suivantes.max(dim=1)[0]
            q_max_suivantes[termines] = 0.0
            q_cibles = recompenses + GAMMA * q_max_suivantes
        
        perte = self.critere(q_actuelles, q_cibles)
        
        self.optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(
            self.reseau_principal.parameters(), max_norm=1.0
        )
        self.optimiseur.step()
        
        self.nb_steps_entrainement += 1
        
        return perte.item()
    
    def fin_episode(self):
        """
        Appelée à la fin de chaque épisode.
        Gère la décroissance d'epsilon et la synchronisation du réseau cible.
        """
        self.nb_episodes += 1
        
        # --- Décroissance d'epsilon ---
        self.epsilon = max(
            EPSILON_FIN,
            self.epsilon * EPSILON_DECROISSANCE
        )
        
        # --- Synchronisation périodique du réseau cible ---
        if self.nb_episodes % MISE_A_JOUR_CIBLE == 0:
            self._synchroniser_cible()
    
    def sauvegarder(self):
        """Sauvegarde le modèle principal."""
        self.reseau_principal.sauvegarder(NOM_MODELE_DQN)
    
    def charger(self):
        """Charge un modèle sauvegardé et synchronise le réseau cible."""
        self.reseau_principal.charger(NOM_MODELE_DQN)
        self._synchroniser_cible()
