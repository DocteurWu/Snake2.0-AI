import time, numpy as np
from marl_game import MARLGame
from marl_agents import DQNAgent, AStarAgent, EconomistAgent, StrategistAgent

COUL = [(80,180,255),(255,60,60),(200,60,200),(50,240,50),(240,220,40),(255,130,0),(40,220,220),(110,110,255)]
jeu = MARLGame(40,40,8)
for i,n in enumerate(["S","P","R","M","E","Ec","St","C"]):
    jeu.ajouter_serpent(i,n,COUL[i])
jeu.initialiser_pommes()
agents = {
 0:DQNAgent("Standard",25,[256,128]),1:DQNAgent("Profond",25,[256,128,64,32]),
 2:DQNAgent("Raycast",16,[256,128]),3:AStarAgent(),4:DQNAgent("Elite",25,[256,128,64,32]),
 5:EconomistAgent(),6:StrategistAgent(),7:DQNAgent("Collectif",25,[256,128,64,32])}
# prefill buffers
d25=(np.zeros(25,np.float32),0,0.0,np.zeros(25,np.float32),False)
for s in [0,1,4,7]:
    for _ in range(20000): agents[s].memoriser(*d25)
d16=(np.zeros(16,np.float32),0,0.0,np.zeros(16,np.float32),False)
for _ in range(20000): agents[2].memoriser(*d16)

def tick(train=True):
    e25={s:jeu.get_vision_grid_state(s) for s in range(8) if jeu.snakes[s].get('actif')}
    ec={}
    for s in range(8):
        if not jeu.snakes[s].get('actif'): continue
        if s in [0,1,4,7]: ec[s]=e25[s]
        elif s==2: ec[s]=jeu.get_raycast_state(s)
    acts={}
    for s in range(8):
        if not jeu.snakes[s].get('actif'): continue
        if s in [0,1,2,4,7]: acts[s]=agents[s].choisir_action(ec[s])
        else: acts[s]=agents[s].choisir_action(jeu,s)
    morts,rew=jeu.step(acts)
    n25={s:jeu.get_vision_grid_state(s) for s in range(8) if jeu.snakes[s].get('actif')}
    nr=jeu.get_raycast_state(2) if jeu.snakes[2].get('actif') else None
    if train:
        for s in [0,1,4,7]:
            agents[s].memoriser(e25[s],acts[s],rew[s],n25[s],(s in morts))
            for _ in range(16): agents[s].entrainer()
        agents[2].memoriser(ec[2],acts[2],rew[2],nr,(2 in morts))
        for _ in range(16): agents[2].entrainer()

# warmup
for _ in range(5): tick(True)
# timing train ON
t0=time.time(); N=30
for _ in range(N): tick(True)
ton=time.time()-t0
# timing train OFF (prefill again not needed; just skip)
t0=time.time()
for _ in range(N): tick(False)
toff=time.time()-t0
print(f"train ON : {ton/N*1000:.1f} ms/tick  ({N/ton:.2f} tick/s)")
print(f"train OFF: {toff/N*1000:.1f} ms/tick  ({N/toff:.2f} tick/s)")
print(f"part entrainement ~ {(ton-toff)/ton*100:.0f}% du temps/tick")
