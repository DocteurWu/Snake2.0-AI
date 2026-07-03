# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Outil de Benchmark et de Suivi de Performance
=============================================================================
 Évalue le modèle sur 100 parties de test et met à jour le graphe du README.
=============================================================================
"""

import os
import json
import time
import matplotlib.pyplot as plt
from game import SnakeGame
from config import TAILLE_ETAT, NB_ACTIONS

HISTORIQUE_PATH = "data/benchmark_history.json"
GRAPH_PATH = "benchmark.png"

def evaluer_modele(agent, nb_parties=100, lock=None):
    """
    Évalue le modèle actuel en mode exploitation pure sur plusieurs parties.
    Utilise la parallélisation par batch pour aller extrêmement vite.
    """
    # 20 jeux en parallèle pour maximiser la vitesse d'inférence
    NB_ENV_TEST = 20
    jeux = [SnakeGame(mode_graphique=False) for _ in range(NB_ENV_TEST)]
    etats = [jeu.reset() for jeu in jeux]
    
    scores = []
    
    while len(scores) < nb_parties:
        # Inférence groupée sans exploration (greedy) avec lock si fourni
        if lock:
            with lock:
                actions = agent.choisir_actions_batch(etats, entrainement=False)
        else:
            actions = agent.choisir_actions_batch(etats, entrainement=False)
        
        for idx in range(NB_ENV_TEST):
            etat_suivant, _, termine, score = jeux[idx].step(actions[idx])
            etats[idx] = etat_suivant
            
            if termine:
                scores.append(score)
                etats[idx] = jeux[idx].reset()
                if len(scores) >= nb_parties:
                    break
                    
    return sum(scores[:nb_parties]) / nb_parties

def enregistrer_et_generer_benchmark(nb_episodes, agent, lock=None):
    """
    Évalue le modèle, enregistre l'historique et génère le graphique log-linéaire.
    """
    try:
        print(f"\n📊 Évaluation du modèle à {nb_episodes} épisodes d'entraînement...")
        score_moyen = evaluer_modele(agent, lock=lock)
        print(f"📈 Score moyen obtenu sur 100 parties de test : {score_moyen:.2f}")
        
        # Charger l'historique existant
        historique = []
        os.makedirs(os.path.dirname(HISTORIQUE_PATH), exist_ok=True)
        if os.path.exists(HISTORIQUE_PATH):
            try:
                with open(HISTORIQUE_PATH, "r", encoding="utf-8") as f:
                    historique = json.load(f)
            except Exception as e:
                print(f"[!] Erreur de lecture de l'historique : {e}. Réinitialisation.")
        
        # Ajouter le nouveau point
        historique.append({
            "episodes": nb_episodes,
            "score_moyen": score_moyen,
            "timestamp": time.time()
        })
        
        # Nettoyer les doublons d'épisodes en gardant la dernière évaluation
        dict_historique = {}
        for entry in historique:
            dict_historique[entry["episodes"]] = entry
        historique = sorted(dict_historique.values(), key=lambda x: x["episodes"])
        
        # Sauvegarder
        with open(HISTORIQUE_PATH, "w", encoding="utf-8") as f:
            json.dump(historique, f, indent=4, ensure_ascii=False)
            
        # Générer le graphique
        generer_graphique(historique)
        
    except Exception as e:
        print(f"[!] Erreur lors de la génération du benchmark : {e}")

def generer_graphique(historique):
    """
    Trace la courbe de performance en échelle logarithmique sur l'axe X.
    """
    if not historique:
        return
        
    episodes = [x["episodes"] for x in historique]
    scores = [x["score_moyen"] for x in historique]
    
    plt.figure(figsize=(10, 5), facecolor="#121212")
    ax = plt.axes()
    ax.set_facecolor("#1e1e1e")
    
    # Tracer la ligne de performance
    plt.plot(episodes, scores, color="#00ffcc", marker="o", linestyle="-", linewidth=2.5, markersize=6, label="Score Moyen (Test)")
    
    # Personnalisation des axes & grille
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines['bottom'].set_color('#333333')
    ax.spines['top'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['right'].set_color('#333333')
    
    plt.title("Évolution des Performances du Serpent IA (Benchmark)", color="white", fontsize=14, pad=15)
    plt.xlabel("Nombre d'épisodes d'entraînement (Échelle Logarithmique)", fontsize=11, labelpad=10)
    plt.ylabel("Score Moyen (Sur 100 parties de test)", fontsize=11, labelpad=10)
    
    # Configurer l'axe X en échelle logarithmique
    plt.xscale("log")
    
    # Grille de repérage
    plt.grid(True, which="both", linestyle="--", alpha=0.2, color="gray")
    
    # Limites et légendes
    plt.ylim(bottom=0)
    plt.legend(loc="upper left", facecolor="#1e1e1e", edgecolor="#333333", labelcolor="white")
    
    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=120, facecolor="#121212")
    plt.close()
    print(f"[OK] Graphique mis à jour : {GRAPH_PATH}")

if __name__ == "__main__":
    # Test autonome du tracé si exécuté directement
    test_data = [
        {"episodes": 10, "score_moyen": 0.5},
        {"episodes": 100, "score_moyen": 2.1},
        {"episodes": 1000, "score_moyen": 8.7},
        {"episodes": 5000, "score_moyen": 14.5},
        {"episodes": 10000, "score_moyen": 22.3}
    ]
    generer_graphique(test_data)
