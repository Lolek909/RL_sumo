import sumo_rl
from stable_baselines3 import DQN

# Tworzenie środowiska RL na podstawie własnej mapy
env = sumo_rl.environment.env.SumoEnvironment(
    net_file="c:/Users/mateu/Sumo/2025-03-25-15-35-18/proste_skrz_sym.net.xml",
    route_file="c:/Users/mateu/Sumo/2025-03-25-15-35-18/my_route.rou.xml",
    use_gui=True,  # Zmień na False, jeśli nie chcesz wizualizacji
    num_seconds=10000,  # Czas trwania symulacji
    yellow_time=3,  # Czas żółtego światła
    min_green=5,  # Minimalny czas zielonego
    max_green=50  # Maksymalny czas zielonego
)

# Tworzenie modelu RL (Deep Q-Network)
model = DQN("MlpPolicy", env, verbose=1)

# Trenowanie modelu
model.learn(total_timesteps=100000)

# Zapis wytrenowanego modelu
model.save("traffic_light_model")

# Testowanie wytrenowanego modelu
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

env.close()
