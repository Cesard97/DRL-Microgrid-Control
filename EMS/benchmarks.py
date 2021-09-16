""""Functions with traditional implementations for the EMS"""
import numpy as np
import pandas as pd
import warnings
from plotly.offline import iplot

from pymgrid.algos import Control
from pymgrid.utils import DataGenerator
from utils.utils import get_metrics

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
                # print(f"Failed to load grid status: {e}")
                grid_is_up = False

            # Define constrains for battery
            p_disc = max(0, min(load - pv, capa_to_dischare, self.microgrid.battery.p_discharge_max))
            p_char = max(0, min(pv - load, capa_to_charge, self.microgrid.battery.p_charge_max))

            control_dict = {}
            # Verify grid status
            if grid_is_up:
                if load >= pv:
                    control_dict = {'battery_charge': 0,
                                    'battery_discharge': p_disc,
                                    'grid_import': max(0, load - pv - p_disc),
                                    'grid_export': 0,
                                    'genset': 0,
                                    'pv_curtailed': 0,
                                    'pv_consummed': min(pv, load),
                                    }
                if load < pv:
                    control_dict = {'battery_charge': p_char,
                                    'battery_discharge': 0,
                                    'grid_import': 0,
                                    'grid_export': max(0, pv - load - p_char),  # abs(min(load-pv,0)),
                                    'genset': 0,
                                    'pv_curtailed': 0,
                                    # 'pv_consummed': min(pv, load+p_char),
                                    'pv_consummed': pv,
                                    }
            else:
                if load >= pv:
                    control_dict = {'battery_charge': 0,
                                    'battery_discharge': p_disc,
                                    'grid_import': 0,
                                    'grid_export': 0,
                                    'genset': max(0, load - pv - p_disc),
                                    'pv_curtailed': 0,
                                    'pv_consummed': min(pv, load),
                                    }
                if load < pv:
                    control_dict = {'battery_charge': p_char,
                                    'battery_discharge': 0,
                                    'grid_import': 0,
                                    'grid_export': 0,  # abs(min(load-pv,0)),
                                    'genset': 0,
                                    'pv_curtailed': max(0, pv - load - p_char),
                                    'pv_consummed': pv,
                                    }

            # Run and save costs
            mg_data = self.microgrid.run(control_dict)
            cost.append(self.microgrid.get_cost())

        # Save results
        # Plot control actions and actual production if requested
        if plot_results:
            self.microgrid.print_control()
            self.microgrid.print_actual_production()

        print("Finish Rule Based Test")
        # Get metrics from production
        metrics_df = get_metrics(self.microgrid)
        # Add costs
        metrics_df['costs'] = cost
        metrics_df.to_csv(f"results/RuleBased/{self.path_prefix}.csv")
        print(f"Results were saved under results/RuleBased/{self.path_prefix} dir")

        return metrics_df

    def run_deterministic_mpc(self, plot_results=False):
        print("Running Deterministic MPC EMS...")
        # Run MPC benchmark on microgrid
        self.microgrid.set_horizon(48)
        self.microgrid.reset(True)

        mpc = Control.ModelPredictiveControl(self.microgrid)
        sample = DataGenerator.return_underlying_data(self.microgrid).iloc[-2650:-1]
        sample.index = np.arange(len(sample))
        output = mpc.run_mpc_on_sample(sample=sample, verbose=True)

        # Get metrics from production
        metrics_df = get_metrics(self.microgrid, output=output)
        # Add costs
        metrics_df['costs'] = output['cost']['cost']

        # Save results
        print("Finish Deterministic MPC Test")
        metrics_df.to_csv(f"results/DeterministicMPC/{self.path_prefix}.csv")
        print(f"Results were saved under results/DeterministicMPC/{self.path_prefix} dir")

        return metrics_df

    def run_saa_mpc(self, plot_results=False):
        print("Running SAA MPC EMS...")
        # Run MPC benchmark on microgrid
        self.microgrid.set_horizon(24)
        self.microgrid.reset(True)
        self.microgrid.benchmarks.run_mpc_benchmark(verbose=True)
        outputs = pd.DataFrame(self.microgrid.benchmarks.outputs_dict)
        cost = np.array(outputs['mpc']['cost']['cost'])
        # Save results
        print("Finish Deterministic SAA MPC Test")
        results = pd.DataFrame.from_dict({'costs': cost})
        results.to_csv(f"results/SAA_MPC/{self.path_prefix}.csv")
        print(f"Results were saved under results/SAA_MPC/{self.path_prefix} dir")

        return results

    def test_all_benchmark_ems(self, plot_results=False):
        """
        Test all implemented benchmarks.
        """
        self.run_rule_based(plot_results)
        self.run_deterministic_mpc(plot_results)
