from cmath import phase
from collections import deque
from numpy import dtype
from dqn_model import DQN_RAM, DQN, CriticNet
import numpy as np
import matplotlib.pyplot as plt
import traci
import torch
from torch import nn
import torch.optim as optim
import os
import torch.nn.functional as F


sumo_cfg_="c:/Users/mateu/Sumo/2025-03-25-15-35-18/proste_skrzyz.sumocfg"

sumo_cmd = ["sumo-gui", "-c", sumo_cfg_]
traci.start(sumo_cmd)

traffic_ligts = traci.trafficlight.getIDList()

print(traffic_ligts)

class Junction:
    # inicjalizacja klasy do kontroli każdego pojedynczego skrzyżowania
    def __init__(self, junction_id, model=None, critic_net=None):
        self.junction_id = junction_id
        self.lanes = traci.trafficlight.getControlledLanes(self.junction_id)  # Pobranie ID pasów
        self.current_phase = 0

        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.junction_id)
        self.num_phases = len(program[0].phases)
        print(f"Skrzyżowanie {self.junction_id}, kontorluje pasy: {self.lanes}")
        print(f"Skrzyżowanie {self.junction_id}, posiada {self.num_phases} faz")
        self.state_history = deque(maxlen=20)
        self.model = model
        self.critic_net = critic_net

    def get_current_state(self):
        return [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]

    def change_phase(self, phase_index):
        if 0 <= phase_index < self.num_phases:
            traci.trafficlight.setPhase(self.junction_id, phase_index)
            self.current_phase = phase_index
            print(phase_index)
        else:
            print("Niepoprawny index")

    def check_the_traffic(self):
        kara = []
        for lane in self.lanes:
            vechicle_count = traci.lane.getLastStepVehicleNumber(lane)
            kara.append(vechicle_count*5) # kara 1
            vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane)
            waiting_time_total = sum(traci.vehicle.getWaitingTime(veh_id)**2 for veh_id in vehicles_on_lane) #kara 2
            kara.append(waiting_time_total)
            speeds = sum(-0.3 * traci.vehicle.getSpeed(veh_id) for veh_id in vehicles_on_lane)
            kara.append(speeds)

        kara = sum(kara)
        nagroda = -kara

        print(f"[INIT] {self.junction_id}, nagroda: {nagroda}")
        return nagroda

    def step(self, use_rl=False):
        state = self.get_current_state()
        nagroda = self.check_the_traffic()
        self.state_history.append(state)
        print(len(self.state_history))
        if len(self.state_history) < self.state_history.maxlen:
            return

        if use_rl and self.model is not None:


            input_tensor = torch.tensor(self.state_history, dtype=torch.float32).flatten().unsqueeze(0) # [1, L*N]
            input_tensor = input_tensor.to(next(self.model.parameters()).device)

            with torch.no_grad():
                predicted_reward = self.critic_net(state)
                q_values = self.model(input_tensor)
                action = torch.argmax(q_values).item()

                loss = F.mse_loss(predicted_reward, torch.tensor(nagroda), dtype=torch.float32)
            self.change_phase(action)
        else:
            self.change_phase(self.current_phase)






    # def print_phase(self):
    #     phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.junction_id)[0].phases
    #     for i, p in enumerate(phases):
    #         if i == self.current_phase:
    #             print(f"Faza {i}: {p.state}")
    #
    # def next_phase(self):
    #     new_phase = (self.current_phase + 1) % self.num_phases
    #     self.change_phase(new_phase)


class TrafficController:
    def __init__(self):
        self.junctions = []
        self._initailize()

    def _initailize(self):
        lights_ids = traci.trafficlight.getIDList()
        for id in lights_ids:
            model = DQN(in_channels=10 * len(traci.trafficlight.getControlledLanes(id)), num_actions=8)
            model_optimizer = optim.Adam(model.parameters(), lr=1e-3)

            critic_net = CriticNet(in_features=len(junction.lanes))
            critic_optimizer = optim.Adam(critic_net.parameters(), lr=1e-3)
            try:
                model.load_state_dict((torch.load("model_skrzyz.pth")))
            except:
                pass
            model = model.cuda()
            model.eval()
            critic_net = CriticNet.cuda()
            critic_net.eval()
            junction = Junction(id, model=model, critic_net=critic_net)
            self.junctions.append(junction)

    def step(self):
        for j in self.junctions:
            # Przykład sterowania: zmiana fazy co 5 kroków
            try:
                j.step(use_rl=True)
            except:
                j.step(use_rl=False)

    def log_reward(self):
        print("rewards for junctions")
        for j in self.junctions:
            reward = j.check_the_traffic()


controller = TrafficController()

step = 0
while step < 1000:
    traci.simulationStep()

    controller.step(step)



    step += 1



