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
import random
from sumolib import checkBinary




def start_sumo(sumo_cfg_="c:/Users/mateu/Sumo/2025-03-25-15-35-18/proste_skrzyz.sumocfg", gui=False):
    if gui:
        sumo_cmd = ["sumo-gui", "-c", sumo_cfg_]
    else:
        sumo_cmd = ["sumo", "-c", sumo_cfg_]
    traci.start(sumo_cmd)



class Junction:
    # inicjalizacja klasy do kontroli każdego pojedynczego skrzyżowania
    def __init__(self, junction_id):
        self.junction_id = junction_id
        self.lanes = traci.trafficlight.getControlledLanes(self.junction_id)  # Pobranie ID pasów
        self.current_phase = 0
        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.junction_id)
        self.num_phases = len(program[0].phases)
        print(f"Skrzyżowanie {self.junction_id}, kontorluje pasy: {self.lanes}")
        print(f"Skrzyżowanie {self.junction_id}, posiada {self.num_phases} faz")
        self.state_history = deque(maxlen=20)

    def get_state(self):
        return [traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]

    def get_num_actions(self):
        return self.num_phases

    def change_phase(self, phase_index):
        traci.trafficlight.setPhase(self.junction_id, phase_index)
        self.current_phase = phase_index


    def get_reward(self):
        kara = []
        for lane in self.lanes:
            vechicle_count = traci.lane.getLastStepVehicleNumber(lane)
            kara.append(vechicle_count)
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



class TrafficController:
    def __init__(self, state_size, action_size, device="cuda"):
        self.junctions = []

        self.policy_net = DQN_RAM(in_features=state_size, num_actions=action_size).to(device)
        self.target_net = DQN_RAM(in_features=state_size, num_actions=action_size).to(device)
        self.critic = CriticNet(in_features=state_size).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.device = device
        self.replay_buffer = deque(maxlen=10)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.policy_net.num_actions)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            print(state_tensor)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def step(self):
        for j in self.junctions:
            # Przykład sterowania: zmiana fazy co 5 kroków
            try:
                j.step(use_rl=True)
            except:
                j.step(use_rl=False)



    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)

        # DQN update
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
        target_q = rewards + self.gamma * max_next_q
        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Critic update
        predicted_reward = self.critic(states)
        critic_loss = F.mse_loss(predicted_reward, rewards)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def save_model(policy_net, target_net, critic, episode, path="models"):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'episode': episode,
        'policy_net': policy_net.state_dict(),
        'target_net': target_net.state_dict(),
        'critic': critic.state_dict(),
    }, os.path.join(path, "checkpoint.pth"))
    print(f"[SAVING] {episode}")

def load_model(policy_net, target_net, critic, path="models/checkpoint.pth"):
    if not os.path.exists(path):
        return 0
    checkpoint = torch.load(path)
    policy_net.load_state_dict(checkpoint['policy_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    critic.load_state_dict(checkpoint['critic'])

    print(f"[LOADING] {checkpoint['episode']}")
    return checkpoint['episode'] + 1

for episode in range(100):
    start_sumo(gui=False)
    for id in traci.trafficlight.getIDList():
        junction = Junction(junction_id=id)
        agent = TrafficController(state_size=len(junction.get_state()), action_size=junction.get_num_actions())

    if episode  == 0:
        start_episode = load_model(policy_net=agent.policy_net, target_net=agent.target_net, critic=agent.critic)



    if episode % 10 == 0:
        save_model(policy_net=agent.policy_net, target_net=agent.target_net, critic=agent.critic, episode=episode)
    for step in range(3000):
        traci.simulationStep()

        state = junction.get_state()
        action = agent.act(state)
        junction.change_phase(action)
        reward = junction.get_reward()
        next_state = junction.get_state()

        agent.remember(state, action, reward, next_state)
        agent.train()

    traci.close()

print("Koniec")
