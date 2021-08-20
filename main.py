from pymgrid import MicrogridGenerator as mgen
from EMS.benchmarks import BenchmarkEms
from EMS.DRL import BaselinesDrlEmsAgents


def get_microgrid_configurations():
    # -----------------------------------------------------
    #              Microgrid Configurations
    # -----------------------------------------------------
    # Create MicroGrid Generator and generate 25 generic microgrids
    generator = mgen.MicrogridGenerator(nb_microgrid=10)
    generator.generate_microgrid()
    generic_microgrids = generator.microgrids
    # Select 3 Generic configurations
    selected_microgrids = [generic_microgrids[0]]

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
        benchmark_ems.test_all_benchmark_ems()

    # Test all DRL EMS
    for i, mg in enumerate(microgrids):
        # Benchmark EMS Object for each mg
        drl_ems = BaselinesDrlEmsAgents(mg, f"{save_dir}-MG{i}")
        drl_ems.test_all_agents()


if __name__ == '__main__':
    # Get microgrid configurations
    microgrids = get_microgrid_configurations()
    # Train All DRL EMS Agents
    train_drl_agents(microgrids, save_dir="TEST00")
    # Test ALL EMS
    test_all_ems(microgrids, save_dir="TEST00")



