# Main libraries
import os
import copy
import random
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import time

# pymigrid
import pymgrid.MicrogridGenerator as mg_gen
import pymgrid.utils.DataGenerator as dg
import pymgrid.algos.Control as control
import pymgrid.Microgrid as mg


# Generate ts for a weak grid
def create_weak_grid_tf(SAIDI=121, SAIFI=120, step_size=0.5):
    # Get prob model from data
    avg_interr_time = SAIDI / SAIFI
    interr_prob = SAIFI / 8760

    # Init full conected grid
    grid_status = np.ones(int(8760 / 0.5))

    for i, gs in enumerate(grid_status):
        # Simulate interruption
        interruption = False

        if random.random() < interr_prob:
            interrup_legnth = random.randint(1, 6)
            grid_status[i:i + interrup_legnth] = 0

    return pd.DataFrame(grid_status, columns=['grid_status'])


def create_microgrid(mg_gen, architecture={'PV':1, 'battery':1, 'genset':0, 'grid':1, 'weak_grid': 0}):
    """
    Creates a microgrid with own data based on specified archtiecture.

    Args:
        mg_gen (PyMigrid.MicroGridGenerator): MicrogridGenerator used for auxiliary functions
        architecture (dict): Microgrid's architecture (1 or 0 for the resource)
    """

    # Random size for the microgrid
    #size_load = np.random.randint(low=100, high=100000)
    size_load = 50000

    # ADD LOAD DATA ----------------------------------------------------
    #r = random.randint(1, 4)
    r = 1
    #load_ts = pd.read_csv(f'./process_data/load{r}.csv')
    load_ts = pd.read_csv(f'./process_data/load_test.csv')
    load = mg_gen._scale_ts(load_ts, size_load, scaling_method='max')
    # ------------------------------------------------------------------

    # Size for all components
    size = mg_gen._size_mg(load, size_load)

    # Intial params
    column_actions=[]
    column_actual_production=[]
    grid_ts=[]
    grid_price_export_ts = []
    grid_price_import_ts = []
    grid_co2_ts = []
    df_parameters = pd.DataFrame()
    df_cost = {'cost':[]}
    df_status = {}
    df_co2 = {'co2':[]}

    # Load
    df_parameters['load'] = [size_load]

    # Costs
    df_parameters['cost_loss_load'] = 10 # 10
    df_parameters['cost_overgeneration'] = 1 # 1
    df_parameters['cost_co2'] = 0.0
    #df_cost['cost'] = [0.0]

    # Status  df
    df_status['load'] = [np.around(load.iloc[0,0],1)]
    df_status['hour'] = [0]
    column_actual_production.append('loss_load')
    column_actual_production.append('overgeneration')
    column_actions.append('load')

    # PV Params
    df_parameters['PV_rated_power'] = np.around(size['pv'], 2)
    column_actual_production.append('pv_consummed')
    column_actual_production.append('pv_curtailed')
    column_actions.append('pv_consummed')
    column_actions.append('pv_curtailed')
    column_actions.append('pv')

    # ADD PV DATA -------------------------------------------------------
    pv_ts = pd.read_csv('./process_data/pv.csv')
    pv = pd.DataFrame(mg_gen._scale_ts(pv_ts, size['pv'], scaling_method='max'))
    # -------------------------------------------------------------------
    df_status['pv'] = [np.around( pv.iloc[0].values[0],1)]

    # Batt Params
    battery = mg_gen._get_battery(capa=size['battery'])
    df_parameters['battery_soc_0'] = battery['soc_0']
    df_parameters['battery_power_charge'] = battery['pcharge']
    df_parameters['battery_power_discharge'] = battery['pdischarge']
    df_parameters['battery_capacity'] = battery['capa']
    df_parameters['battery_efficiency'] = battery['efficiency']
    df_parameters['battery_soc_min'] = battery['soc_min']
    df_parameters['battery_soc_max'] = battery['soc_max']
    df_parameters['battery_cost_cycle'] = battery['cost_cycle']
    column_actual_production.append('battery_charge')
    column_actual_production.append('battery_discharge')
    column_actions.append('battery_charge')
    column_actions.append('battery_discharge')
    df_status['battery_soc'] = [battery['soc_0']]

    capa_to_charge = max(
        (df_parameters['battery_soc_max'].values[0] * df_parameters['battery_capacity'].values[0] -
          df_parameters['battery_soc_0'].iloc[-1] *
          df_parameters['battery_capacity'].values[0]
          ) / df_parameters['battery_efficiency'].values[0], 0)

    capa_to_discharge = max((df_parameters['battery_soc_0'].iloc[-1] *
                              df_parameters['battery_capacity'].values[0]
                              - df_parameters['battery_soc_min'].values[0] *
                              df_parameters['battery_capacity'].values[0])
                              * df_parameters['battery_efficiency'].values[0], 0)

    df_status['capa_to_charge'] = [np.around(capa_to_charge,1)]
    df_status['capa_to_discharge'] = [np.around(capa_to_discharge,1)]

    grid_spec=0

    if architecture['grid']==1:

        #rand_weak_grid = np.random.randint(low=0, high=2)
        #rand_weak_grid = 0
        #price_scenario = np.random.randint(low=1, high=3)

        # Weak grid
        if architecture['weak_grid'] == 1:
          # Genset for weak grid
          architecture['genset'] = 1
          # Get weak grid
          grid_ts =create_weak_grid_tf()
        else:
          # Not weak grid
          grid_ts = pd.DataFrame(np.ones(int(17520)), columns=['grid_status'])

        #grid = self._get_grid(rated_power=size['grid'], weak_grid=rand_weak_grid, price_scenario=price_scenario)

        df_parameters['grid_weak'] = architecture['weak_grid']
        df_parameters['grid_power_import'] = size['grid']
        df_parameters['grid_power_export'] = size['grid']

        column_actual_production.append('grid_import')
        column_actual_production.append('grid_export')
        column_actions.append('grid_import')
        column_actions.append('grid_export')
        df_status['grid_status'] = [grid_ts.iloc[0,0]]


        # Ignored Co2 time series (zeros)
        grid_co2_ts = pd.DataFrame(np.zeros((17520,)))

        df_status['grid_co2'] = [grid_co2_ts.iloc[0, 0]]

        # ADD GRID DATA --------------------------------------------------------
        grid_tariff = 0.15

        grid_price_import_ts = pd.DataFrame(np.ones((17520,))*grid_tariff)
        grid_price_export_ts = pd.DataFrame(np.ones((17520,))*grid_tariff*0.4)
        # ----------------------------------------------------------------------

        df_status['grid_price_import'] = [grid_price_import_ts.iloc[0,0]]
        df_status['grid_price_export'] = [grid_price_export_ts.iloc[0,0]]

    # TO DO: GENSET ARCHITECTURE
    if architecture['genset']==1:
        genset = mg_gen._get_genset(rated_power=size['genset'])
        df_parameters['genset_polynom_order'] = len(genset['polynom'])

        for i in range(len(genset['polynom'])):
            df_parameters['genset_polynom_'+str(i)]=genset['polynom'][i]

        df_parameters['genset_rated_power'] = genset['rated_power']
        df_parameters['genset_pmin'] = genset['pmin']
        df_parameters['genset_pmax'] = genset['pmax']
        #df_parameters['fuel_cost'] = genset['fuel_cost']
        df_parameters['fuel_cost'] = 0.5
        df_parameters['genset_co2'] = genset['co2']
        column_actual_production.append('genset')
        column_actions.append('genset')


    df_actions= {i:[] for i in column_actions} #pd.DataFrame(columns = column_actions, )
    df_actual_production = {i:[] for i in column_actual_production} #pd.DataFrame(columns=column_actual_production)

    microgrid_spec={
        'parameters':df_parameters, #Dictionary
        'df_actions':df_actions, #Dataframe
        'architecture':architecture, #Dictionary
        'df_status':df_status, #Dictionary
        'df_actual_generation':df_actual_production,#Dataframe
        'grid_spec':grid_spec, #value = 0
        'df_cost':df_cost, #Dataframe of 1 value = 0.0
        'df_co2': df_co2,
        'pv':pv, #Dataframe
        'load': load, #Dataframe
        'grid_ts':grid_ts, #Dataframe
        'control_dict': column_actions, #dictionnary
        'grid_price_import' : grid_price_import_ts,
        'grid_price_export' : grid_price_export_ts,
        'grid_co2': grid_co2_ts,
    }

    microgrid = mg.Microgrid(microgrid_spec, horizon=24, timestep=0.5)

    return microgrid


class MicroGridEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, microgrid, prev_steps=12, seed=42):
        super(MicroGridEnv, self).__init__()
        # Set seed
        np.random.seed(seed)
        # Set Microgrid
        self.mg = microgrid
        self.mg.train_test_split(train_size=0.8)
        # Set Action Space: 0-Batt, 1-Grid, 2-Genset
        self.action_space = gym.spaces.Box(np.array([-1, -1, 0]), np.array([+1, +1, +1]), dtype=np.float64)
        # Set Observation Space
        self.prev_steps = prev_steps
        self.Ns = len(self.mg._df_record_state.keys())
        self.observation_space = gym.spaces.Box(low=-1, high=np.float('inf'), shape=(self.Ns,), dtype=np.float64)
        # Set step
        self.step_count = 0
        self.done = False

    def step(self, action):

        # RUN ACTION ON THE MICROGRID

        # Continous actions
        control_dict = self.get_action_continuous(action)
        self.mg.run(control_dict)

        # UPDATE STATE AND REWARD
        # self.state = self.transition()
        # State
        new_state = self.mg.get_updated_values()
        self.state = list(new_state.values())
        # Reward
        self.reward = self.get_reward()
        # Other info
        self.done = self.mg.done
        self.info = {}
        self.step_count += 1

        # WARNINGS -----------------------------------------------------------------
        # if self.done:
        #    print("WARNING : EPISODE DONE")  # should never reach this point
        #    return self.state, self.reward, self.done, self.info
        try:
            assert (self.observation_space.contains(self.state))
        except AssertionError:
            print("ERROR : INVALID STATE", self.state)
        try:
            assert (self.action_space.contains(action))
        except AssertionError:
            print("ERROR : INVALD ACTION", action)
        # --------------------------------------------------------------------------

        return self.state, self.reward, self.done, self.info

    def get_reward(self):
        # get cost
        r = -self.mg.get_cost()

        return r

        # Mapping between action and the control_dict

    def get_action_continuous(self, action):
        """
        :param action: current action
        :return: control_dict : dicco of controls
        """
        '''
        Actions are:
        binary variable whether charging or dischargin
        battery power, normalized to 1
        binary variable whether importing or exporting
        grid power, normalized to 1
        binary variable whether genset is on or off
        genset power, normalized to 1
        '''
        # GET STATE OF THE GRID
        mg = self.mg

        # PV
        # pv = mg.pv
        pv = mg.get_updated_values()['pv']

        # Load
        # load = mg.load
        load = mg.get_updated_values()['load']
        net_load = load - pv

        # Batt charge
        capa_to_charge = mg.battery.capa_to_charge
        p_charge_max = mg.battery.p_charge_max
        p_charge = max(0, min(-net_load, capa_to_charge, p_charge_max))
        # Batt discharge
        capa_to_discharge = mg.battery.capa_to_discharge
        p_discharge_max = mg.battery.p_discharge_max
        p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))

        # BUILD ACTION
        control_dict = {}

        # Battery action
        if mg.architecture['battery'] == 1:
            control_dict['battery_charge'] = max(0, min(action[0] * mg.battery.capacity,
                                                        mg.battery.capa_to_charge,
                                                        mg.battery.p_charge_max))

            control_dict['battery_discharge'] = max(0, min((1 - action[0]) * mg.battery.capacity,
                                                           mg.battery.capa_to_discharge,
                                                           mg.battery.p_discharge_max))
        # Grid Action
        if mg.architecture['grid'] == 1:
            if mg.grid.status == 1:
                control_dict['grid_import'] = max(0, min(action[1] * mg.grid.power_import,
                                                         mg.grid.power_import))
                control_dict['grid_export'] = max(0, min((1 - action[1]) * mg.grid.power_export,
                                                         mg.grid.power_export))
            else:
                # avoid warnings
                control_dict['grid_import'] = 0
                control_dict['grid_export'] = 0

        # TO DO: Genset action
        if mg.architecture['genset'] == 1:
            control_dict['genset'] = max(0, min(action[2] * mg.genset.rated_power,
                                                mg.genset.rated_power))

        # PV action
        # control_dict['pv_consummed'] = min(pv, load + control_dict['battery_charge'])

        return control_dict

    def reset(self, use_test_split=False):
        # Reset microgrid
        self.mg.reset(use_test_split)
        # Rest count
        self.step_count = 0
        # Get new state
        new_state = self.mg.get_updated_values()
        obs = list(new_state.values())
        self.done = False
        return obs
