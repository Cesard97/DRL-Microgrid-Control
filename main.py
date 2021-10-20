from pymgrid import MicrogridGenerator as mgen
from EMS.benchmarks import BenchmarkEms
from EMS.DRL import BaselinesDrlEmsAgents
from results.result_plotter import ResultsPlotter

import pandas as pd
import time
import numpy as np
pd.set_option('display.max_columns', 20)


def get_microgrid_configurations():
    # -----------------------------------------------------
    #              Microgrid Configurations
    # -----------------------------------------------------
    # Create MicroGrid Generator and generate 25 generic microgrids
    generator = mgen.MicrogridGenerator(nb_microgrid=10)
    generator.generate_microgrid()
    generic_microgrids = generator.microgrids

    # Select 6 Generic configurations
    selected_microgrids = [generic_microgrids[9]] + generic_microgrids[1:5] + [generic_microgrids[6]]

    # Colombian Configurations
    # TODO

    return selected_microgrids


def train_drl_agents(microgrids, save_dir="TEST00"):
    # Train for all microgrids
    for i, mg in enumerate(microgrids):
        # DRL EMS Agent for each mg
        ems_agents = BaselinesDrlEmsAgents(mg, f"{save_dir}-MG{i}")
        # Train all Agents
        ems_agents.train_all_agents()


def test_ems(microgrids, save_dir="TEST00"):
    # Test all DRL EMS
    for i, mg in enumerate(microgrids):
        # DRL EMS Object for each mg
        drl_ems = BaselinesDrlEmsAgents(mg, f"{save_dir}-MG{i}")
        drl_ems.test_all_agents()

    # Test all benchmark EMS
    for i, mg in enumerate(microgrids):
        # Benchmark EMS Object for each mg
        benchmark_ems = BenchmarkEms(mg, f"{save_dir}-MG{i}")
        benchmark_ems.test_all_benchmark_ems()


def optimize_parameters(microgrid, save_dir="Hyper_Opt"):
    drl_ems = BaselinesDrlEmsAgents(microgrid, save_dir)
    drl_ems.optimize_parameters()


if __name__ == '__main__':

    # Get microgrid configurations
    microgrids = get_microgrid_configurations()

    # Train All DRL EMS Agents
    # train_drl_agents(microgrids, save_dir="TEST01")

    # Optimize parameters
    optimize_parameters(microgrids[0])

    # Test ALL EMS
    # start = time.time()
    # test_ems(microgrids, save_dir="TEST01")
    # end = time.time()
    # print("TESTING TIME:")
    # print(f"{end-start} seconds")

    # Plot Results
    # plotter = ResultsPlotter('TEST01')
    # plotter.plot_training()
    # plotter.plot_costs()
    # plotter.plot_battery_usage()

    # Print summary tables
    # plotter.print_costs_table(normalize=True)
    # plotter.print_renewable_use_table()
    # plotter.print_batt_cycle_table()
    # plotter.print_power_derivative_table(metric='max')

