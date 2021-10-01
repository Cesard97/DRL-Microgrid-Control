import os

import pandas as pd
from copy import copy
from stable_baselines.bench import Monitor
from utils.SaveCallback import SaveOnBestTrainingRewardCallback
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv as CsDaMicroGridEnv
from pymgrid.Environments.pymgrid_csca import ContinuousMicrogridEnv, SafeExpMicrogridEnv
from utils.utils import get_metrics

from stable_baselines import DQN, PPO2, A2C, TD3


class BaselinesDrlEmsAgents:

    def __init__(self, microgrid, path_prefix='TEST00', time_steps=100000):
        # Main Attributes
        self.time_steps = time_steps

        # Create log dir
        self.dir_DQN = f"DQN/{path_prefix}"
        self.dir_dis_PPO = f"PPO_Discrete/{path_prefix}"
        self.dir_con_PPO = f"PPO_Continuous/{path_prefix}"
        self.dir_dis_A2C = f"A2C_Discrete/{path_prefix}"
        self.dir_con_A2C = f"A2C_Continuous/{path_prefix}"
        self.dir_TD3 = f"TD3/{path_prefix}"

        os.makedirs(f"models/{self.dir_DQN}", exist_ok=True)
        os.makedirs(f"models/{self.dir_dis_PPO}", exist_ok=True)
        os.makedirs(f"models/{self.dir_con_PPO}", exist_ok=True)
        os.makedirs(f"models/{self.dir_dis_A2C}", exist_ok=True)
        os.makedirs(f"models/{self.dir_con_A2C}", exist_ok=True)
        os.makedirs(f"models/{self.dir_TD3}", exist_ok=True)

        # Create discrete gym environment for microgrid
        self.microgrid = copy(microgrid)
        self.continuous_microgrid = copy(microgrid)
        self.microgrid.train_test_split(train_size=0.7)

        self.discrete_env = CsDaMicroGridEnv({'microgrid': self.microgrid,
                                              'forecast_args': None,
                                              'resampling_on_reset': False,
                                              'baseline_sampling_args': None})
        self.continuous_env = SafeExpMicrogridEnv(self.continuous_microgrid)

    # ---------------------------- TRAINING ----------------------------

    def train_dqn_ems(self):
        print(f"Training DQN Based EMS for {self.time_steps} steps...")
        # Create env
        env = Monitor(self.discrete_env, f"models/{self.dir_DQN}")
        env.reset(testing=False)
        # Create Model
        model = DQN('MlpPolicy', env, double_q=True, exploration_fraction=0.25, verbose=0)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=6000,
                                                    log_dir=f"models/{self.dir_DQN}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        return model

    def train_discrete_ppo_ems(self):
        print(f"Training Discrete PPO Based EMS for {self.time_steps} steps...")
        # Create env
        env = Monitor(self.discrete_env, f"models/{self.dir_dis_PPO}")
        env.reset(testing=False)
        # Create A2C Model
        model = PPO2('MlpPolicy', env, verbose=0)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=6000,
                                                    log_dir=f"models/{self.dir_dis_PPO}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        return model

    def train_continuous_ppo_ems(self):
        print(f"Training Continuous PPO Based EMS for {self.time_steps} steps...")
        env = Monitor(self.continuous_env, f"models/{self.dir_con_PPO}")
        env.reset()
        # Create A2C Model
        model = PPO2('MlpPolicy', env, verbose=0, n_steps=1024, learning_rate=0.001)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=6000,
                                                    log_dir=f"models/{self.dir_con_PPO}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps*5, callback=callback)

        return model

    def train_discrete_a2c_ems(self):
        print(f"Training Discrete A2C Based EMS for {self.time_steps} steps...")
        # Create env
        env = Monitor(self.discrete_env, f"models/{self.dir_dis_A2C}")
        env.reset(testing=False)
        # Create A2C Model
        model = A2C('MlpPolicy', env, verbose=0)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=6000,
                                                    log_dir=f"models/{self.dir_dis_A2C}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        return model

    def train_continuous_a2c_ems(self):
        print(f"Training Continuous A2C Based EMS for {self.time_steps} steps...")
        env = Monitor(self.continuous_env, f"models/{self.dir_con_A2C}")
        env.reset()
        # Create A2C Model
        model = A2C('MlpPolicy', env, verbose=0)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=6000,
                                                    log_dir=f"models/{self.dir_con_A2C}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps*5, callback=callback)

        return model

    def train_td3_ems(self):
        print(f"Training Continuous TD3 Based EMS for {self.time_steps} steps...")
        env = Monitor(self.continuous_env, f"models/{self.dir_TD3}")
        env.reset()
        # Create A2C Model
        model = TD3('MlpPolicy', env, verbose=0)
        # Create the callback: check every 100 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=6000,
                                                    log_dir=f"models/{self.dir_TD3}")
        # Train Agent
        model.learn(total_timesteps=self.time_steps*5, callback=callback)

        return model

    def train_all_agents(self):
        """
        Train all agents with implemented algorithms.
        """
        # Continuous Env
        self.train_continuous_ppo_ems()
        self.train_continuous_a2c_ems()
        self.train_td3_ems()

        # Discrete env
        self.train_dqn_ems()
        self.train_discrete_ppo_ems()
        self.train_discrete_a2c_ems()

    # ---------------------------- TESTING ----------------------------

    def test_dqn_ems(self):
        print(f"Testing DQN Based EMS...")
        # Load best DQN Agent
        model = DQN.load(f"models/{self.dir_DQN}/best_model.zip")
        cost = []
        # Test DQN Agent
        obs = self.discrete_env.reset(testing=True)
        while not self.discrete_env.done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = self.discrete_env.step(action)
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
        obs = self.discrete_env.reset(testing=True)
        while not self.discrete_env.done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = self.discrete_env.step(action)
            cost.append(-rewards)

        # Get metrics from production
        metrics_df = get_metrics(self.microgrid)
        # Add costs
        metrics_df['costs'] = cost
        # Save results
        metrics_df.to_csv(f"results/{self.dir_dis_PPO}.csv")

        return metrics_df

    def test_continuous_ppo_ems(self):
        print(f"Testing continuous PPO Based EMS...")
        # Load best PPO Agent
        model = PPO2.load(f"models/{self.dir_con_PPO}/best_model.zip")
        # Test best PPO Agent
        cost = []
        obs = self.continuous_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = self.continuous_env.step(action)
            cost.append(-rewards)

        # Get metrics from production
        metrics_df = get_metrics(self.continuous_env.microgrid)
        # Add costs
        metrics_df['costs'] = cost
        # Save results
        metrics_df.tail(2628).to_csv(f"results/{self.dir_con_PPO}.csv")

        return metrics_df

    def test_discrete_a2c_ems(self):
        print(f"Testing discrete PPO Based EMS...")
        # Load best PPO Agent
        model = A2C.load(f"models/{self.dir_dis_A2C}/best_model.zip")
        # Test best PPO Agent
        cost = []
        obs = self.discrete_env.reset(testing=True)
        while not self.discrete_env.done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = self.discrete_env.step(action)
            cost.append(-rewards)

        # Get metrics from production
        metrics_df = get_metrics(self.microgrid)
        # Add costs
        metrics_df['costs'] = cost
        # Save results
        metrics_df.to_csv(f"results/{self.dir_dis_A2C}.csv")

        return metrics_df

    def test_continuous_a2c_ems(self):
        print(f"Testing continuous PPO Based EMS...")
        # Load best PPO Agent
        model = A2C.load(f"models/{self.dir_con_A2C}/best_model.zip")
        # Test best PPO Agent
        cost = []
        obs = self.continuous_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = self.continuous_env.step(action)
            cost.append(-rewards)

        # Get metrics from production
        metrics_df = get_metrics(self.continuous_env.microgrid)
        # Add costs
        metrics_df['costs'] = cost
        # Save results
        metrics_df.tail(2628).to_csv(f"results/{self.dir_con_A2C}.csv")

        return metrics_df

    def test_td3_ems(self):
        print(f"Testing TD3 Based EMS...")
        # Load best PPO Agent
        model = TD3.load(f"models/{self.dir_TD3}/best_model.zip")
        # Test best PPO Agent
        cost = []
        obs = self.continuous_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = self.continuous_env.step(action)
            cost.append(-rewards)

        # Get metrics from production
        metrics_df = get_metrics(self.continuous_env.microgrid)
        # Add costs
        metrics_df['costs'] = cost
        # Save results
        metrics_df.tail(2628).to_csv(f"results/{self.dir_TD3}.csv")

        return metrics_df

    def test_all_agents(self):
        """
        Test all implemented agents.
        """
        # Test Discrete EMS
        self.test_dqn_ems()
        self.test_discrete_ppo_ems()
        self.test_discrete_ppo_ems()
        self.test_discrete_a2c_ems()

        # Test Continuous EMS
        self.test_continuous_a2c_ems()
        self.test_continuous_ppo_ems()
        self.test_td3_ems()
