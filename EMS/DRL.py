import os

import pandas as pd

from stable_baselines.bench import Monitor
from utils.SaveCallback import SaveOnBestTrainingRewardCallback
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv as CsDaMicroGridEnv

from stable_baselines import DQN, PPO2


class BaselinesDrlEmsAgents:

    def __init__(self, microgrid, path_prefix='TEST00', time_steps=30000):
        # Main Attributes
        self.microgrid = microgrid
        self.time_steps = time_steps

        # Create log dir
        self.dir_DQN = f"DQN/{path_prefix}"
        self.dir_dis_PPO = f"DiscretePPO/{path_prefix}"

        os.makedirs(f"models/{self.dir_DQN}", exist_ok=True)
        os.makedirs(f"models/{self.dir_dis_PPO}", exist_ok=True)

        # Create discrete gym environment for microgrid
        discrete_env = CsDaMicroGridEnv({'microgrid': microgrid,
                                         'forecast_args': None,
                                         'resampling_on_reset': False,
                                         'baseline_sampling_args': None})

        # Create environment and monitors for each algorithm
        self.env_DQN = Monitor(discrete_env, f"models/{self.dir_DQN}")
        self.env_dis_PPO = Monitor(discrete_env, f"models/{self.dir_dis_PPO}")

    # ---------------------------- TRAINING ----------------------------

    def train_dqn_ems(self):
        print(f"Training DQN Based EMS for {self.time_steps} steps...")
        self.env_DQN.reset()
        # Create Model
        model = DQN('MlpPolicy', self.env_DQN, double_q=True, exploration_fraction=0.25, verbose=1)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=int(self.time_steps/10),
                                                    log_dir=f"models/{self.dir_DQN}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        return model

    def train_discrete_ppo_ems(self):
        print(f"Training Discrete PPO Based EMS for {self.time_steps} steps...")
        self.env_dis_PPO.reset()
        # Create A2C Model
        model = PPO2('MlpPolicy', self.env_dis_PPO, verbose=1)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=int(self.time_steps/10),
                                                    log_dir=f"models/{self.dir_dis_PPO}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        return model

    def train_all_agents(self):
        """
        Train all implemented agents.
        """
        self.train_dqn_ems()
        self.train_discrete_ppo_ems()

    # ---------------------------- TESTING ----------------------------

    def test_dqn_ems(self):
        print(f"Testing DQN Based EMS...")
        # Load best DQN Agent
        model = DQN.load(f"models/{self.dir_DQN}/best_model.zip")
        cost = []
        # Test DQN Agent
        obs = self.env_DQN.reset(True)
        while not self.env_DQN.done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = self.env_DQN.step(action)
            cost.append(-rewards)

        # Save results
        results = pd.DataFrame.from_dict({'costs': cost})
        results.to_csv(f"results/{self.dir_DQN}.csv")

        return results

    def test_discrete_ppo_ems(self):
        print(f"Testing discrete PPO Based EMS...")
        # Load best PPO Agent
        model = PPO2.load(f"models/{self.dir_dis_PPO}/best_model.zip")
        # Test best PPO Agent
        cost = []
        obs = self.env_dis_PPO.reset(True)
        while not self.env_dis_PPO.done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = self.env_dis_PPO.step(action)
            cost.append(-rewards)

        # Save results
        results = pd.DataFrame.from_dict({'costs': cost})
        results.to_csv(f"results/{self.dir_dis_PPO}.csv")

        return results

    def test_all_agents(self):
        """
        Test all implemented agents.
        """
        self.test_dqn_ems()
        self.test_discrete_ppo_ems()
