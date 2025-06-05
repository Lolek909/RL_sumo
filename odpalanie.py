from stable_baselines3 import PPO
import traci
import gym
import numpy as np


# Załaduj model PPO
model = PPO.load("ppo_sumo.zip")



sumo_cfg_="c:/Users/mateu/Sumo/2025-03-25-15-35-18/proste_skrzyz.sumocfg"
junction_id_ = "9917802298"


class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg=sumo_cfg_, junction_id=junction_id_):
        super(SumoEnv, self).__init__()
        self.sumo_cfg = sumo_cfg
        self.junction_id = junction_id
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

    def reset(self):
        if traci.isLoaded():  # Sprawdza, czy SUMO jest już uruchomione
            traci.close()
        traci.start(["sumo", "-c", self.sumo_cfg])
        self.lanes = traci.trafficlight.getControlledLanes(self.junction_id)  # Pobranie ID pasów
        print(f"Pasy kontrolowane przez {self.junction_id}: {self.lanes}")  # Debugging
        return np.array([0])

    def step(self, action):
        if action == 0:
            traci.trafficlight.setPhase(self.junction_id, 0)
        else:
            traci.trafficlight.setPhase(self.junction_id, 1)

        traci.simulationStep()

        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in
                         self.lanes]  # Pobranie korków ze wszystkich pasów
        total_queue = sum(queue_lengths)

        reward = -total_queue
        done = traci.simulation.getMinExpectedNumber() == 0
        return np.array([total_queue]), reward, done, {}

    def close(self):
        if traci.isLoaded(): traci.close()
