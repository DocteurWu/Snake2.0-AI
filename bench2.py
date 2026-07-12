import time, numpy as np, torch
import marl_agents
from marl_game import MARLGame
from marl_agents import DQNAgent, AStarAgent, EconomistAgent, StrategistAgent

print("AMP dispo:", marl_agents._USE_AMP, "| threads:", torch.get_num_threads())

COUL=[(80,180,255),(255,60,60),(200,60,200),(50,240,50),(240,220,40),(255,130,0),(40,220,220),(110,110,255)]
jeu=MARLGame(40,40,8)
for i,n in enumerate(["S","P","R","M","E","Ec","St","C"]): jeu.ajouter_serpent(i,n,COUL[i])
jeu.initialiser_pommes()
agents={0:DQNAgent("Standard",25,[256,128]),1:DQNAgent("Profond",25,[256,128,64,32]),
 2:DQNAgent("Raycast",16,[256,128]),3:AStarAgent(),4:DQNAgent("Elite",25,[256,128,64,32]),
 5:EconomistAgent(),6:StrategistAgent(),7:DQNAgent("Collectif",25,[256,128,64,32])}
d25=(np.zeros(25,np.float32),0,0.0,np.zeros(25,np.float32),False)
for s in [0,1,4,7]:
    for _ in range(20000): agents[s].memoriser(*d25)
for _ in range(20000): agents[2].memoriser(np.zeros(16,np.float32),0,0.0,np.zeros(16,np.float32),False)

N=20
# warmup
for _ in range(5):
    e25={s:jeu.get_vision_grid_state(s) for s in range(8)}
    ec={s:e25[s] for s in [0,1,4,7]}; ec[2]=jeu.get_raycast_state(2)
    acts={s:agents[s].choisir_action(ec[s]) for s in [0,1,2,4,7]}
    for s in [3,5,6]: acts[s]=agents[s].choisir_action(jeu,s)
    m,r=jeu.step(acts)
    n25={s:jeu.get_vision_grid_state(s) for s in range(8)}
    for s in [0,1,4,7]: agents[s].memoriser(e25[s],acts[s],r[s],n25[s],(s in m)); agents[s].entrainer()
    agents[2].memoriser(ec[2],acts[2],r[2],jeu.get_raycast_state(2),(2 in m)); agents[2].entrainer()

t0=time.time()
for _ in range(N):
    e25={s:jeu.get_vision_grid_state(s) for s in range(8)}
    ec={s:e25[s] for s in [0,1,4,7]}; ec[2]=jeu.get_raycast_state(2)
    acts={s:agents[s].choisir_action(ec[s]) for s in [0,1,2,4,7]}
    for s in [3,5,6]: acts[s]=agents[s].choisir_action(jeu,s)
    m,r=jeu.step(acts)
    n25={s:jeu.get_vision_grid_state(s) for s in range(8)}
    for s in [0,1,4,7]: agents[s].memoriser(e25[s],acts[s],r[s],n25[s],(s in m))
    for _ in range(8): agents[s].entrainer() if s in [0,1,4,7] else None
    agents[2].memoriser(ec[2],acts[2],r[2],jeu.get_raycast_state(2),(2 in m))
    for _ in range(8): agents[2].entrainer()
dt=time.time()-t0
print(f"NOUVEAU (AMP+8train) : {dt/N*1000:.1f} ms/tick  ({N/dt:.2f} tick/s)")
print(f"vs baseline 3057 ms/tick -> speedup x{dt/N*1000/3057:.2f}")
