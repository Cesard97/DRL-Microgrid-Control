import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

from stable_baselines.results_plotter import load_results, ts2xy


class ResultsPlotter:

    def __init__(self, path_prefix='TEST00'):
        self.path_prefix = path_prefix
        # Get all results in results folder
        self.results_dirs = os.listdir('./results')
        self.results_dirs.remove('result_plotter.py')
        self.results_dirs.remove('__pycache__')

        self.model_dirs = os.listdir('./models')

    def plot_training(self):
        # Plot data
        fig = plt.figure(figsize=(10, 7))

        for d in self.model_dirs:
            x, y = ts2xy(load_results(f"./models/{d}/{self.path_prefix}/"), 'timesteps')
            plt.plot(x, y)

        # Labels
        plt.title(f'Soften reward curve on training ({self.path_prefix})', fontsize=20)
        plt.xlabel('Step', fontsize=15)
        plt.ylabel('Reward (USD)', fontsize=15)
        plt.legend(self.model_dirs, fontsize=12)
        plt.grid(True)
        plt.show()

    def plot_test(self, metric='costs'):
        # Plot data
        fig = plt.figure(figsize=(10, 7))
        for d in self.results_dirs:
            data = pd.read_csv(f"./results/{d}/{self.path_prefix}.csv")[metric]
            plt.plot(np.cumsum(data))

        # Labels
        plt.title(f'Cumulative Operational Cost ({self.path_prefix})', fontsize=20)
        plt.xlabel('Step', fontsize=15)
        plt.ylabel('USD', fontsize=15)
        plt.legend(self.results_dirs, fontsize=12)
        plt.grid(True)
        plt.show()

    def print_summary_table(self):
        pass


if __name__ == '__main__':
    plotter = ResultsPlotter('TEST00-MG0')
    plotter.plot_training()
    plotter.plot_test()
