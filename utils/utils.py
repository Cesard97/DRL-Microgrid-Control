import pandas as pd
import numpy as np


def get_metrics(microgrid, output=None):
    # Get production and control df from output
    if output != None:
        prod_df = pd.DataFrame.from_dict(output['production'])
        control_df = pd.DataFrame.from_dict(output['action'])
        print(prod_df.head())
        print(control_df.head())
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

    metrics = pd.DataFrame.from_dict({"pv_consumed": pv_consumed, "pv": pv, "batt_cycles": batt_cycles})
    return metrics
