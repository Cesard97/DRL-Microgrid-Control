# Main Import
import os
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv as CsDaMicroGridEnv

# Stable baselines
from stable_baselines import DQN, PPO2, A2C

# Create log dir
log_dir_DA_DQN = "tmp/DA_DQN/"
log_dir_DA_A2C = "tmp/DA_A2C/"
log_dir_CA_A2C = "tmp/CA_A2C/"
log_dir_DA_PPO = "tmp/DA_PPO/"
log_dir_CA_PPO = "tmp/CA_PPO/"

os.makedirs(log_dir_DA_DQN, exist_ok=True)
os.makedirs(log_dir_DA_A2C, exist_ok=True)
os.makedirs(log_dir_CA_A2C, exist_ok=True)
os.makedirs(log_dir_DA_PPO, exist_ok=True)
os.makedirs(log_dir_CA_PPO, exist_ok=True)

# CREATE ENVIROMENT

# Wrap env to monitor
mg_da_env = CsDaMicroGridEnv({'microgrid':microgrid_weak, 'forecast_args':None, 'resampling_on_reset':False, 'baseline_sampling_args':None})
da_env_DQN = Monitor(mg_da_env, log_dir_DA_DQN)

# ca_env_A2C = Monitor(mg_ca_env, log_dir_CA_A2C)
da_env_A2C = Monitor(mg_da_env, log_dir_DA_A2C)
# ca_env_PPO = Monitor(mg_ca_env, log_dir_CA_PPO)
da_env_PPO = Monitor(mg_da_env, log_dir_DA_PPO)

# Training Steps
time_steps = 250000

# ---------------------------------------------------------
#                   Stable Baselines
# ---------------------------------------------------------

# DISCRETE ACTION SPACE

# DQN
da_env_DQN.reset()
# Create Model
da_model_DQN = DQN('MlpPolicy', da_env_DQN, double_q=True, exploration_fraction=0.25, verbose=1)
# Create the callback: check every 100 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=3000, log_dir=log_dir_DA_DQN)
# Train the agent
da_model_DQN.learn(total_timesteps=time_steps, callback=callback)

# A2C
da_env_A2C.reset()
# Create A2C Model
da_model_A2C = A2C('MlpPolicy', da_env_A2C, verbose=1)
# Create the callback: check every 100 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=3000, log_dir=log_dir_DA_A2C)
# Train Agent
da_model_A2C.learn(total_timesteps=time_steps, callback=callback)

# PPO
da_env_PPO.reset()
# Create A2C Model
da_model_PPO = PPO2('MlpPolicy', da_env_PPO, verbose=1)
# Create the callback: check every 100 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=3000, log_dir=log_dir_DA_PPO)
# Train Agent
da_model_PPO.learn(total_timesteps=time_steps, callback=callback)
