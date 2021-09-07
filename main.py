from pymgrid import MicrogridGenerator as mgen
from EMS.benchmarks import BenchmarkEms
from EMS.DRL import BaselinesDrlEmsAgents
from results.result_plotter import ResultsPlotter

import pandas as pd
pd.set_option('display.max_columns', 20)


def get_microgrid_configurations():
    # -----------------------------------------------------
    #              Microgrid Configurations
    # -----------------------------------------------------
    # Create MicroGrid Generator and generate 25 generic microgrids
    generator = mgen.MicrogridGenerator(nb_microgrid=10)
    generator.generate_microgrid()
    generic_microgrids = generator.microgrids
    # Select 3 Generic configurations
    selected_microgrids = generic_microgrids[0:7]

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


def test_all_ems(microgrids, save_dir="TEST00"):

    # Test all benchmark EMS
    for i, mg in enumerate(microgrids):
        # Benchmark EMS Object for each mg
        benchmark_ems = BenchmarkEms(mg, f"{save_dir}-MG{i}")
        # benchmark_ems.test_all_benchmark_ems()
        benchmark_ems.run_deterministic_mpc()

    # Test all DRL EMS
    for i, mg in enumerate(microgrids):
        # Benchmark EMS Object for each mg
        # drl_ems = BaselinesDrlEmsAgents(mg, f"{save_dir}-MG{i}")
        # drl_ems.test_all_agents()
        pass


if __name__ == '__main__':

    plotter = ResultsPlotter('TEST00')
    # Get microgrid configurations
    microgrids = get_microgrid_configurations()
    # Train All DRL EMS Agents
    # train_drl_agents(microgrids, save_dir="TEST00")
    # Test ALL EMS
    test_all_ems(microgrids, save_dir="TEST00")
    # Plot Results
    plotter.plot_training()
    # plotter.plot_test()
    plotter.print_summary_table(normalize=True)




