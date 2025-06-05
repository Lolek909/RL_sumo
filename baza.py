import os
import traci
import numpy as np
import gym
from build.lib.traci import junction
from gym import spaces
from stable_baselines3 import PPO

sumo_cfg_="c:/Users/mateu/Sumo/2025-03-25-15-35-18/proste_skrzyz.sumocfg"
junction_id_ = "9917802298"


class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg=sumo_cfg_, junction_id=junction_id_):
        super(SumoEnv, self).__init__()
        self.sumo_cfg = sumo_cfg
        self.junction_id = junction_id
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

    def reset(self):
        if traci.isLoaded():  # Sprawdza, czy SUMO jest już uruchomione
            traci.close()
        traci.start(["sumo-gui", "-c", self.sumo_cfg])


        self.lanes = traci.trafficlight.getControlledLanes(self.junction_id)  # Pobranie ID pasów
        print(f"Pasy kontrolowane przez {self.junction_id}: {self.lanes}")  # Debugging
        return np.array([0])

    def step(self, action):
        if action == 0:
            traci.trafficlight.setPhase(self.junction_id, 0)
        elif action == 1:
            traci.trafficlight.setPhase(self.junction_id, 1)
        elif action == 2:
            traci.trafficlight.setPhase(self.junction_id, 2)
        elif action == 3:
            traci.trafficlight.setPhase(self.junction_id, 3)
        elif action == 4:
            traci.trafficlight.setPhase(self.junction_id, 4)
        elif action == 5:
            traci.trafficlight.setPhase(self.junction_id, 5)
        elif action == 6:
            traci.trafficlight.setPhase(self.junction_id, 6)
        else:
            traci.trafficlight.setPhase(self.junction_id, 7)

        traci.simulationStep()

        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in
                         self.lanes]  # Pobranie korków ze wszystkich pasów
        total_queue = sum(queue_lengths)

        reward = -total_queue
        done = traci.simulation.getMinExpectedNumber() == 0
        return np.array([total_queue]), reward, done, {}

    def close(self):
        if traci.isLoaded():
            traci.close()

def model_training():
    env = SumoEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("ppo_sumo")
    obs = env.reset()

    env.close()
model_training()

def model_eval():
    model = PPO.load("ppo_sumo")

    env = SumoEnv()
    obs = env.reset()

    # Symulujemy ruch przez 1000 kroków
    for _ in range(1000):
        action, _ = model.predict(obs)  # Model przewiduje akcję
        obs, reward, done, _ = env.step(action)  # Wykonujemy akcję

        if done:
            break

    env.close()

model_eval()