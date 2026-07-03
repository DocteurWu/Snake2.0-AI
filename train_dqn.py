# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Entraînement Headless DQN Haute Performance
=============================================================================
 Mode sans affichage graphique (pure ligne de commande) optimisé au maximum.
 Charge le modèle existant s'il existe pour continuer l'apprentissage.
 Sauvegarde automatique à l'arrêt ou toutes les 100 parties.
=============================================================================
"""

import os
import sys
import time
import signal
import argparse
import torch
import numpy as np

from config import (
    TAILLE_GRILLE, VITESSE_JEU,
    TAILLE_ETAT, NB_ACTIONS, NOM_MODELE_DQN, CHEMIN_MODELE
)
from game import SnakeGame
from agent_dqn import AgentDQN

# Variables globales pour la gestion propre de l'arrêt
interrompu = False
agent = None

def signal_handler(sig, frame):
    global interrompu
    print("\n[!] Signal d'arrêt reçu. Sauvegarde du modèle en cours...")
    interrompu = True

# Enregistrer les signaux d'interruption (Ctrl+C, arrêt de tâche)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    global agent, interrompu
    
    parser = argparse.ArgumentParser(description="Entraînement DQN Headless")
    parser.add_argument("--max-steps-s", type=int, default=0, 
                        help="Nombre maximum de steps de simulation par seconde (0 = débridé/vitesse maximale)")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" SNAKE IA — Entraînement Headless DQN Haute Performance")
    print("=" * 60)
    if args.max_steps_s > 0:
        print(f"[*] Limite de vitesse configurée : {args.max_steps_s} steps/s")
    else:
        print("[*] Limite de vitesse : Vitesse MAX débridée")
    print("=" * 60)
    
    agent = AgentDQN()
    
    # Charger le modèle existant s'il existe
    chemin_complet = os.path.join(CHEMIN_MODELE, NOM_MODELE_DQN)
    if os.path.exists(chemin_complet):
        try:
            agent.charger()
            print(f"[OK] Reprise de l'apprentissage depuis {chemin_complet}")
        except Exception as e:
            print(f"[!!] Erreur lors du chargement de {chemin_complet} ({e}). Démarrage de zéro.")
    else:
        print("[*] Aucun modèle existant trouvé. Démarrage de zéro.")
        
    # Hyper-optimisation ARM : Nombre d'environnements parallèles en arrière-plan
    NB_ENVS = 32
    print(f"[*] Initialisation de {NB_ENVS} environnements de simulation...")
    jeux = [SnakeGame(mode_graphique=False) for _ in range(NB_ENVS)]
    etats = [jeu.reset() for jeu in jeux]
    
    scores_recents = []
    meilleur_score_moyen = 0.0
    total_episodes = 0
    total_steps = 0
    
    dernier_log_temps = time.time()
    dernier_log_steps = 0
    
    print("\n🚀 Entraînement démarré (Appuyez sur Ctrl+C pour arrêter et sauvegarder)\n")
    print(f"{'Épisodes':<10} | {'Moyenne (100)':<14} | {'Meilleur Moy':<14} | {'Epsilon':<9} | {'Vitesse':<15}")
    print("-" * 75)
    
    try:
        while not interrompu:
            temps_debut_batch = time.time()
            
            # 1. Choisir les actions en batch pour tous les environnements
            actions = agent.choisir_actions_batch(etats, entrainement=True)
            
            # 2. Exécuter un step sur chaque environnement
            for idx in range(NB_ENVS):
                etat_actuel = etats[idx]
                action = actions[idx]
                
                etat_suivant, recompense, termine, score = jeux[idx].step(action)
                agent.memoriser(etat_actuel, action, recompense, etat_suivant, termine)
                
                etats[idx] = etat_suivant
                total_steps += 1
                
                # 3. Entraîner l'agent toutes les 4 étapes cumulées
                if total_steps % 4 == 0:
                    agent.entrainer()
                
                # 4. Gérer la fin de partie d'un serpent
                if termine:
                    scores_recents.append(score)
                    if len(scores_recents) > 100:
                        scores_recents.pop(0)
                        
                    total_episodes += 1
                    agent.fin_episode()
                    etats[idx] = jeux[idx].reset()
                    
                    # Sauvegarde périodique et benchmark
                    if total_episodes % 100 == 0:
                        agent.sauvegarder()
                        from benchmark import enregistrer_et_generer_benchmark
                        enregistrer_et_generer_benchmark(total_episodes, agent)
            
            # Limiter la vitesse si demandé
            if args.max_steps_s > 0:
                duree_calculee = time.time() - temps_debut_batch
                duree_cible = NB_ENVS / args.max_steps_s
                temps_sommeil = duree_cible - duree_calculee
                if temps_sommeil > 0:
                    time.sleep(temps_sommeil)
                        
            # 5. Affichage régulier des statistiques (toutes les 3 secondes pour ne pas ralentir le CPU)
            temps_actuel = time.time()
            if temps_actuel - dernier_log_temps >= 3.0:
                duree = temps_actuel - dernier_log_temps
                vitesse = int((total_steps - dernier_log_steps) / duree)
                
                moyenne_actuelle = sum(scores_recents) / len(scores_recents) if scores_recents else 0.0
                if moyenne_actuelle > meilleur_score_moyen and len(scores_recents) >= 50:
                    meilleur_score_moyen = moyenne_actuelle
                
                print(f"{total_episodes:<10} | {moyenne_actuelle:<14.2f} | {meilleur_score_moyen:<14.2f} | {agent.epsilon:<9.3f} | {vitesse:<6} steps/s")
                
                dernier_log_temps = temps_actuel
                dernier_log_steps = total_steps
                
    except KeyboardInterrupt:
        pass
    finally:
        # Assurer la sauvegarde finale
        if agent is not None:
            agent.sauvegarder()
        print("\n[OK] Modèle sauvegardé avec succès. Entraînement terminé.")

if __name__ == "__main__":
    main()
