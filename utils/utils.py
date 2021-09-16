import pandas as pd
import numpy as np


def get_metrics(microgrid, output=None):
    # Get production and control df from output
    if output is not None:
        prod_df = pd.DataFrame.from_dict(output['production'])
        control_df = pd.DataFrame.from_dict(output['action'])
    else:
        # Get production and control df from microgrid
        prod_df = pd.DataFrame.from_dict(microgrid._df_record_actual_production)
        control_df = pd.DataFrame.from_dict(microgrid._df_record_control_dict)

    # Get % of used PV
    pv_consumed = np.array(prod_df['pv_consummed'])
    pv = np.array(control_df['pv'])

    # Get % of batt cycles
    batt_capacity = microgrid.battery.capacity
    batt_char = prod_df['battery_charge']
    batt_dis = prod_df['battery_discharge']
    batt_cycles = (np.array(batt_dis) - np.array(batt_char))/batt_capacity

    # Get Grid Behaviour
    try:
        # Get rated grid power
        grid_power = microgrid.grid.power_import
        # Get grid curve
        grid_import = prod_df['grid_import']
        grid_export = prod_df['grid_export']
        grid_curve = (np.array(grid_import) - np.array(grid_export)) / grid_power

    except:
        # Return zeros if there is no grid
        grid_curve = np.zeros(len(pv))

    metrics = pd.DataFrame.from_dict({"pv_consumed": pv_consumed,
                                      "pv": pv,
                                      "batt_cycles": batt_cycles,
                                      "grid_curve": grid_curve})
    return metrics
