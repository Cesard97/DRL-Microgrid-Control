import os

import pandas as pd

from stable_baselines.bench import Monitor
from utils.SaveCallback import SaveOnBestTrainingRewardCallback
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv as CsDaMicroGridEnv
from pymgrid.Environments.pymgrid_csca import ContinuousMicrogridEnv
from utils.utils import get_metrics

from stable_baselines import DQN, PPO2, A2C


class BaselinesDrlEmsAgents:

    def __init__(self, microgrid, path_prefix='TEST00', time_steps=60000):
        # Main Attributes
        self.time_steps = time_steps

        # Create log dir
        self.dir_DQN = f"DQN/{path_prefix}"
        self.dir_dis_PPO = f"DiscretePPO/{path_prefix}"
        self.dir_con_PPO = f"ContinuousPPO/{path_prefix}"
        self.dir_dis_A2C = f"DiscreteA2C/{path_prefix}"

        os.makedirs(f"models/{self.dir_DQN}", exist_ok=True)
        os.makedirs(f"models/{self.dir_dis_PPO}", exist_ok=True)
        os.makedirs(f"models/{self.dir_con_PPO}", exist_ok=True)
        os.makedirs(f"models/{self.dir_dis_A2C}", exist_ok=True)

        # Create discrete gym environment for microgrid
        self.microgrid = microgrid
        self.microgrid.train_test_split(train_size=0.7)
        discrete_env = CsDaMicroGridEnv({'microgrid': self.microgrid,
                                         'forecast_args': None,
                                         'resampling_on_reset': False,
                                         'baseline_sampling_args': None})

        continuous_env = ContinuousMicrogridEnv({'microgrid': self.microgrid,
                                                 'forecast_args': None,
                                                 'resampling_on_reset': False,
                                                 'baseline_sampling_args': None})

        # Create environment and monitors for each algorithm
        self.env_DQN = Monitor(discrete_env, f"models/{self.dir_DQN}")
        self.env_dis_PPO = Monitor(discrete_env, f"models/{self.dir_dis_PPO}")
        self.env_con_PPO = Monitor(continuous_env, f"models/{self.dir_con_PPO}")
        self.env_dis_A2C = Monitor(discrete_env, f"models/{self.dir_dis_A2C}")

    # ---------------------------- TRAINING ----------------------------

    def train_dqn_ems(self):
        print(f"Training DQN Based EMS for {self.time_steps} steps...")
        self.env_DQN.reset(testing=False)
        # Create Model
        model = DQN('MlpPolicy', self.env_DQN, double_q=True, exploration_fraction=0.25, verbose=0)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=5000,
                                                    log_dir=f"models/{self.dir_DQN}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        return model

    def train_discrete_ppo_ems(self):
        print(f"Training Discrete PPO Based EMS for {self.time_steps} steps...")
        self.env_dis_PPO.reset(testing=False)
        # Create A2C Model
        model = PPO2('MlpPolicy', self.env_dis_PPO, verbose=0)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=5000,
                                                    log_dir=f"models/{self.dir_dis_PPO}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        return model

    def train_continuous_ppo_ems(self):
        print(f"Training Continuous PPO Based EMS for {self.time_steps} steps...")
        self.env_con_PPO.reset(testing=False)
        # Create A2C Model
        model = PPO2('MlpPolicy', self.env_con_PPO, verbose=0)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=5000,
                                                    log_dir=f"models/{self.dir_con_PPO}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        return model

    def train_discrete_a2c_ems(self):
        print(f"Training Discrete A2C Based EMS for {self.time_steps} steps...")
        self.env_dis_A2C.reset(testing=False)
        # Create A2C Model
        model = A2C('MlpPolicy', self.env_dis_A2C, verbose=0)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=5000,
                                                    log_dir=f"models/{self.dir_dis_A2C}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        return model

    def train_all_agents(self):
        """
        Train all implemented agents.
        """
        self.train_dqn_ems()
        self.train_discrete_ppo_ems()
        self.train_continuous_ppo_ems()

    # ---------------------------- TESTING ----------------------------

    def test_dqn_ems(self):
        print(f"Testing DQN Based EMS...")
        # Load best DQN Agent
        model = DQN.load(f"models/{self.dir_DQN}/best_model.zip")
        cost = []
        # Test DQN Agent
        obs = self.env_DQN.reset(testing=True)
        while not self.env_DQN.done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = self.env_DQN.step(action)
            cost.append(-rewards)

        # Get metrics from production
        metrics_df = get_metrics(self.microgrid)
        # Add costs
        metrics_df['costs'] = cost
        # Save results
        metrics_df.to_csv(f"results/{self.dir_DQN}.csv")

        return metrics_df

    def test_discrete_ppo_ems(self):
        print(f"Testing discrete PPO Based EMS...")
        # Load best PPO Agent
        model = PPO2.load(f"models/{self.dir_dis_PPO}/best_model.zip")
        # Test best PPO Agent
        cost = []
        obs = self.env_dis_PPO.reset(testing=True)
        while not self.env_dis_PPO.done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = self.env_dis_PPO.step(action)
            cost.append(-rewards)

        # Get metrics from production
        metrics_df = get_metrics(self.microgrid)
        # Add costs
        metrics_df['costs'] = cost
        # Save results
        metrics_df.to_csv(f"results/{self.dir_dis_PPO}.csv")

        return metrics_df

    def test_all_agents(self):
        """
        Test all implemented agents.
        """
        self.test_dqn_ems()
        self.test_discrete_ppo_ems()
