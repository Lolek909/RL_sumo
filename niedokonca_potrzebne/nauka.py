import traci
sumo_cfg_="c:/Users/mateu/Sumo/2025-03-25-15-35-18/proste_skrzyz.sumocfg"

sumo_cmd = ["sumo-gui", "-c", sumo_cfg_]
traci.start(sumo_cmd)

tl_id = traci.trafficlight.getIDList()[0] # sprawdzamy ile mamy faz do obsługi pierwszego skrzyżowania
phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].phases



for i, p in enumerate(phases):
    print(f"Faza {i}: {p.state}") # drukujemy fazy wszystkie fazy przypisane do danego skrzyżowania

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (zainstalowana):", torch.version.cuda)

