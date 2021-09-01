""""Functions with traditional implementations for the EMS"""
import numpy as np
import pandas as pd
import warnings
from plotly.offline import iplot

from pymgrid.algos import Control
from pymgrid.utils import DataGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)


class BenchmarkEms:

    def __init__(self, microgrid, path_prefix='TEST00'):
        # Attributes
        self.microgrid = microgrid
        self.microgrid.train_test_split(train_size=0.7)
        self.path_prefix = path_prefix

    def run_rule_based(self, plot_results=False):
        print("Running Rule Based EMS...")
        self.microgrid.reset(True)
        cost = []
        # Get initial state of the microgrid
        mg_data = self.microgrid.get_updated_values()

        while not self.microgrid.done:
            # Get microgrid data
            load = mg_data['load']
            pv = mg_data['pv']
            capa_to_charge = mg_data['capa_to_charge']
            capa_to_dischare = mg_data['capa_to_discharge']
            try:
                grid_is_up = mg_data['grid_status']
            except Exception as e:
                print(f"Failed to load grid status: {e}")
                grid_is_up = False

            # Define constrains for battery
            p_disc = max(0, min(load - pv, capa_to_dischare, self.microgrid.battery.p_discharge_max))
            p_char = max(0, min(pv - load, capa_to_charge, self.microgrid.battery.p_charge_max))

            # Verify grid status
            if grid_is_up:
                if load >= pv:
                    control_dict = {'battery_charge': 0,
                                    'battery_discharge': p_disc,
                                    'grid_import': max(0, load - pv - p_disc),
                                    'grid_export': 0,
                                    'genset': 0,
                                    'pv_curtailed': 0,
                                    'pv_consumed': min(pv, load),
                                    }
                if load < pv:
                    control_dict = {'battery_charge': p_char,
                                    'battery_discharge': 0,
                                    'grid_import': 0,
                                    'grid_export': max(0, pv - load - p_char),  # abs(min(load-pv,0)),
                                    'genset': 0,
                                    'pv_curtailed': 0,
                                    # 'pv_consummed': min(pv, load+p_char),
                                    'pv_consumed': pv,
                                    }
            else:
                if load >= pv:
                    control_dict = {'battery_charge': 0,
                                    'battery_discharge': p_disc,
                                    'grid_import': 0,
                                    'grid_export': 0,
                                    'genset': max(0, load - pv - p_disc),
                                    'pv_curtailed': 0,
                                    'pv_consumed': min(pv, load),
                                    }
                if load < pv:
                    control_dict = {'battery_charge': p_char,
                                    'battery_discharge': 0,
                                    'grid_import': 0,
                                    'grid_export': 0,  # abs(min(load-pv,0)),
                                    'genset': 0,
                                    'pv_curtailed': max(0, pv - load - p_char),
                                    'pv_consumed': pv,
                                    }

            # you call run passing it your control_dict and it will return the mg_data for the next timestep
            mg_data = self.microgrid.run(control_dict)
            cost.append(self.microgrid.get_cost())

        # Save results
        print("Finish Rule Based Test")
        results = pd.DataFrame.from_dict({'costs': cost})
        results.to_csv(f"results/RuleBased/{self.path_prefix}.csv")
        print(f"Results were saved under results/RuleBased/{self.path_prefix} dir")
        # Plot control actions and actual production if requested
        if plot_results:
            self.microgrid.print_control()
            self.microgrid.print_actual_production()

        return results

    def run_deterministic_mpc(self, plot_results=False):
        print("Running Deterministic MPC EMS...")
        # Run MPC benchmark on microgrid
        self.microgrid.set_horizon(24)
        self.microgrid.reset(True)

        mpc = Control.ModelPredictiveControl(self.microgrid)
        sample = DataGenerator.return_underlying_data(self.microgrid).iloc[-2650:-1]
        sample.index = np.arange(len(sample))
        output = mpc.run_mpc_on_sample(sample=sample, verbose=True)
        cost = output['cost']['cost']

        # Save results
        print("Finish Deterministic MPC Test")
        results = pd.DataFrame.from_dict({'costs': cost})
        results.to_csv(f"results/DeterministicMPC/{self.path_prefix}.csv")
        print(f"Results were saved under results/DeterministicMPC/{self.path_prefix} dir")
        if plot_results:
            fig1 = output.iplot(asFigure=True)
            iplot(fig1)

        return results

    def run_deterministic_mpc_2(self, plot_results=False):
        print("Running Deterministic MPC EMS...")
        # Run MPC benchmark on microgrid
        self.microgrid.set_horizon(24)
        self.microgrid.reset(True)
        self.microgrid.benchmarks.run_mpc_benchmark(verbose=True)
        outputs = pd.DataFrame(self.microgrid.benchmarks.outputs_dict)
        cost = np.array(outputs['mpc']['cost']['cost'])
        print(outputs.keys())

        # Save results
        print("Finish Deterministic MPC Test")
        results = pd.DataFrame.from_dict({'costs': cost})
        results.to_csv(f"results/DeterministicMPC/{self.path_prefix}.csv")
        print(f"Results were saved under results/DeterministicMPC/{self.path_prefix} dir")

        return results

    def test_all_benchmark_ems(self, plot_results=True):
        """
        Test all implemented benchmarks.
        """
        self.run_rule_based(plot_results)
        self.run_deterministic_mpc(plot_results)
