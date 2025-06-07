from cmath import phase
from collections import deque
from numpy import dtype
from dqn_model import DQN_RAM, DQN_RAM, CriticNet
import numpy as np
import matplotlib.pyplot as plt
import traci
import torch
from torch import nn
import torch.optim as optim
import os
import torch.nn.functional as F
import random


sumo_cfg_="c:/Users/mateu/Sumo/2025-03-25-15-35-18/proste_skrzyz.sumocfg"

sumo_cmd = ["sumo-gui", "-c", sumo_cfg_]
traci.start(sumo_cmd)

traffic_ligts = traci.trafficlight.getIDList()

class Junction:
    def __init__(self, junction_id):
        self.junction_id = junction_id
        self.lanes = traci.trafficlight.getControlledLanes(junction_id)
        self.num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(junction_id)[0].phases)
        self.current_phase = 0

    def get_state(self):
        return np.array([traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes], dtype=np.float32)

    def apply_action(self, phase_index):
        traci.trafficlight.setPhase(self.junction_id, phase_index)
        self.current_phase = phase_index

    def get_reward(self):
        reward = 0.0
        for lane in self.lanes:
            v = traci.lane.getLastStepVehicleNumber(lane)
            reward -= v * 5
            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                reward -= traci.vehicle.getWaitingTime(veh_id)**2
                reward -= 0.3 * traci.vehicle.getSpeed(veh_id)
        return reward

    def get_num_actions(self):
        return self.num_phases

class RLAgent:
    def __init__(self, state_size, action_size, device="cuda"):
        self.policy_net = DQN_RAM(in_channels=state_size, num_actions=action_size).to(device)
        self.target_net = DQN_RAM(in_channels=state_size, num_actions=action_size).to(device)
        self.critic = CriticNet(in_features=state_size).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.device = device
        self.replay_buffer = deque(maxlen=100)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.2
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.policy_net.num_actions)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

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



for id in traci.trafficlight.getIDList():
    junction = Junction(junction_id=id)
    agent = RLAgent(state_size=len(junction.get_state()), action_size=junction.get_num_actions())

for step in range(10000):
    traci.simulationStep()

    state = junction.get_state()
    action = agent.act(state)
    junction.apply_action(action)
    reward = junction.get_reward()
    next_state = junction.get_state()

    agent.remember(state, action, reward, next_state)
    agent.train()



traci.trafficlight.getIDList()

