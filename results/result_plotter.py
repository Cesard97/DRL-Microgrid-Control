import os
import time
import optuna
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

from stable_baselines.results_plotter import load_results, ts2xy
from optuna import visualization


class ResultsPlotter:

    def __init__(self, path_prefix='TEST00'):
        # Attributes
        self.path_prefix = path_prefix
        self.N = 6
        # Get all results in results folder
        self.results_dirs = os.listdir('./results')
        self.results_dirs.remove('result_plotter.py')
        self.results_dirs.remove('__pycache__')
        self.results_dirs.remove('summary')
        self.model_dirs = os.listdir('./models')

    def plot_training(self):
        for i in range(0, self.N):
            # Plot data
            fig = plt.figure(figsize=(12, 8))

            for d in self.model_dirs:
                x, y = ts2xy(load_results(f"./models/{d}/{self.path_prefix}-MG{i}/"), 'timesteps')
                plt.plot(x, y)

            # Labels
            plt.title(f'Soften reward curve on training ({self.path_prefix}-MG{i})', fontsize=20)
            plt.xlabel('Step', fontsize=15)
            plt.ylabel('Reward (USD)', fontsize=15)
            plt.legend(self.model_dirs, fontsize=12)
            plt.grid(True)
            plt.show()

    def plot_costs(self):
        for i in range(0, self.N):
            # Plot data
            fig = plt.figure(figsize=(12, 8))
            for d in self.results_dirs:
                data = np.cumsum(pd.read_csv(f"./results/{d}/{self.path_prefix}-MG{i}.csv")['costs'])
                plt.plot(data)

            # Labels
            plt.title(f'Cumulative Operational Cost ({self.path_prefix}-MG{i})', fontsize=20)
            plt.xlabel('Step', fontsize=15)
            plt.ylabel('USD', fontsize=15)
            plt.legend(self.results_dirs, fontsize=12)
            plt.grid(True)
            plt.show()

    def plot_battery_usage(self):
        for i in range(0, self.N):
            # Plot data
            fig = plt.figure(figsize=(12, 8))
            for d in self.results_dirs:
                data = pd.read_csv(f"./results/{d}/{self.path_prefix}-MG{i}.csv")['batt_cycles']
                plt.plot(data)

            # Labels
            plt.title(f'Normalized Battery Usage ({self.path_prefix}-MG{i})', fontsize=20)
            plt.xlabel('Step', fontsize=15)
            plt.ylabel('Normalized Power', fontsize=15)
            plt.legend(self.results_dirs, fontsize=12)
            plt.grid(True)
            plt.show()

    def print_costs_table(self, normalize=True):
        # Pandas Dataframe
        results_df = pd.DataFrame.from_dict({'Model': self.results_dirs})
        # Loop over mg
        for i in range(0, self.N):
            data = []
            # Loop over models
            for r in self.results_dirs:
                total_cost = np.sum(pd.read_csv(f"./results/{r}/{self.path_prefix}-MG{i}.csv")['costs'].values)
                data.append(np.round(total_cost, 3))

            results_df[f'MG-{i}'] = data

        # Normalize against DeterministicMPC
        if normalize:
            for i in range(0, self.N):
                results_df[f'MG-{i}'] = results_df[f'MG-{i}'].div(results_df.iloc[3][f'MG-{i}'])

        # Add Avg. Column
        if normalize:
            results_df['Mean'] = results_df.drop('Model', axis=1).mean(axis=1)
            print("TOTAL OPERATIONAL COSTS (USD)")
        else:
            results_df['Sum'] = results_df.drop('Model', axis=1).sum(axis=1)
            print("TOTAL OPERATIONAL COSTS (Normalized respect to MPC)")
        # Print results df
        results_df.to_csv(f"./results/summary/costs-{self.path_prefix}.csv", sep=';')
        print(results_df)

    def print_renewable_use_table(self):
        # Pandas Dataframe
        results_df = pd.DataFrame.from_dict({'Model': self.results_dirs})
        # Loop over mg
        for i in range(0, self.N):
            data = []
            # Loop over models
            for r in self.results_dirs:
                pv_consumed = np.sum(pd.read_csv(f"./results/{r}/{self.path_prefix}-MG{i}.csv")['pv_consumed'].values)
                pv = np.sum(pd.read_csv(f"./results/{r}/{self.path_prefix}-MG{i}.csv")['pv'].values)
                used_pv = pv_consumed/pv
                data.append(np.round(used_pv, 3))

            results_df[f'MG-{i}'] = data

        results_df['Mean'] = results_df.drop('Model', axis=1).mean(axis=1)

        # Print results df
        print('% OF RENEWABLE USAGE')
        results_df.to_csv(f"./results/summary/renewable-{self.path_prefix}.csv", sep=';')
        print(results_df)

    def print_batt_cycle_table(self):
        # Pandas Dataframe
        results_df = pd.DataFrame.from_dict({'Model': self.results_dirs})
        # Loop over mg
        for i in range(0, self.N):
            data = []
            # Loop over models
            for r in self.results_dirs:
                batt_cycles = np.abs(pd.read_csv(f"./results/{r}/{self.path_prefix}-MG{i}.csv")['batt_cycles'].values)
                batt_cycles = np.sum(batt_cycles)
                data.append(np.round(batt_cycles, 3))

            results_df[f'MG-{i}'] = data

        results_df['Mean'] = results_df.drop('Model', axis=1).mean(axis=1)

        # Print results df
        print('EQUIVALENT BATTERY CYCLES')
        results_df.to_csv(f"./results/summary/batt_cycles-{self.path_prefix}.csv", sep=';')
        print(results_df)

    def print_power_derivative_table(self, metric='avg'):
        # Pandas Dataframe
        results_df = pd.DataFrame.from_dict({'Model': self.results_dirs})
        # Loop over mg
        for i in range(0, self.N):
            data = []
            # Loop over models
            for r in self.results_dirs:
                grid_curve = np.abs(pd.read_csv(f"./results/{r}/{self.path_prefix}-MG{i}.csv")['grid_curve'].values)
                if metric == 'max':
                    power_diff = np.max(np.diff(grid_curve))
                else:
                    power_diff = np.mean(np.diff(grid_curve))
                data.append(power_diff)

            results_df[f'MG-{i}'] = data

        results_df['Mean'] = results_df.drop('Model', axis=1).mean(axis=1)
        # Print results df
        print('POWER GRID CYCLES')
        results_df.to_csv(f"./results/summary/grid_cycles-{self.path_prefix}.csv", sep=';')
        print(results_df)

    def plot_hyperparameter_optimization(self, rl_algorithm='DQN'):
        # Load study
        study = optuna.load_study(study_name=f"{rl_algorithm}_Opt", storage="sqlite:///Hyper_Opt.db")
        # Print best results
        print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
        # Plot history
        visualization.plot_optimization_history(study)
        # Plot param importance
        visualization.plot_param_importances(study)


if __name__ == '__main__':
    plotter = ResultsPlotter('TEST00-MG0')
    plotter.plot_training()
    plotter.plot_costs()
