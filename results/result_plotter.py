import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ResultsPlotter:

    def __init__(self, path_prefix='TEST00'):
        self.path_prefix = path_prefix

    def plot_training(self):
        pass

    def plot_test(self, metric='costs'):
        # Get all results in results folder
        dirs = os.listdir('.')
        dirs.remove('result_plotter.py')

        # Plot data
        fig = plt.figure(figsize=(10, 7))
        for d in dirs:
            data = pd.read_csv(f"{d}/{self.path_prefix}.csv")[metric]
            if d == "DeterministicMPC":
                data = data/100
            plt.plot(np.cumsum(data))

        # Labels
        plt.title('Cumulative Operational Cost (Weak Microgrid)', fontsize=20)
        plt.xlabel('Step', fontsize=15)
        plt.ylabel('USD', fontsize=15)
        plt.legend(dirs, fontsize=12)
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    plotter = ResultsPlotter('TEST00-MG0')
    plotter.plot_test()
