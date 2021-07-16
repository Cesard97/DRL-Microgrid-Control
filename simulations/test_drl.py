# Stable baselines
from stable_baselines import DQN, PPO2, A2C

# ---------------------------------------------------------
#                         TEST
# ---------------------------------------------------------

# Load best DQN Agent
model_DQN = DQN.load("tmp/DQN/best_model.zip")
# Test DQN Agent
DQN_cost = []
obs = mg_da_env.reset(True)
while not mg_da_env.done:
    action, _states = da_model_DQN.predict(obs)
    obs, rewards, dones, info = mg_da_env.step(action)
    DQN_cost.append(-rewards)


# Load best A2C Agent
da_model_A2C = A2C.load("tmp/A2C/best_model.zip")
# Test A2C Agent
da_A2C_cost = []
obs = mg_da_env.reset(True)
while not mg_da_env.done:
    action, _states = da_model_A2C.predict(obs)
    obs, rewards, dones, info = mg_da_env.step(action)
    da_A2C_cost.append(-rewards)

# Load best PPO Agent
da_model_PPO = PPO2.load("tmp/PPO/best_model.zip")
# Test best PPO Agent
da_PPO_cost = []
obs = mg_da_env.reset(True)
while not mg_da_env.done:
    action, _states = da_model_PPO.predict(obs)
    obs, rewards, dones, info = mg_da_env.step(action)
    da_PPO_cost.append(-rewards)

# ---------------------------------------------------------
#                   Export Results
# ---------------------------------------------------------

