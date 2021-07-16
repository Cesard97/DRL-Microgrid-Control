
def run_rule_based(microgrid):
    costs = []
    # Get initial state of the microgrid
    mg_data = microgrid.get_updated_values()

    while not microgrid.done:
        # Get microgrid data
        load = mg_data['load']
        pv = mg_data['pv']
        capa_to_charge = mg_data['capa_to_charge']
        capa_to_dischare = mg_data['capa_to_discharge']
        try:
            grid_is_up = mg_data['grid_status']
        except:
            grid_is_up = False

        # Define constrains for battery
        p_disc = max(0, min(load - pv, capa_to_dischare, microgrid.battery.p_discharge_max))
        p_char = max(0, min(pv - load, capa_to_charge, microgrid.battery.p_charge_max))

        # Verify grid status
        if grid_is_up:
            if load - pv >= 0:
                control_dict = {'battery_charge': 0,
                                'battery_discharge': p_disc,
                                'grid_import': max(0, load - pv - p_disc),
                                'grid_export': 0,
                                'genset': 0,
                                'pv_consummed': min(pv, load),
                                }
            if load - pv < 0:
                control_dict = {'battery_charge': p_char,
                                'battery_discharge': 0,
                                'grid_import': 0,
                                'grid_export': max(0, pv - load - p_char),  # abs(min(load-pv,0)),
                                'genset': 0,
                                # 'pv_consummed': min(pv, load+p_char),
                                'pv_consummed': pv,
                                }
        else:
            if load - pv >= 0:
                control_dict = {'battery_charge': 0,
                                'battery_discharge': p_disc,
                                'grid_import': 0,
                                'grid_export': 0,
                                'genset': max(0, load - pv - p_disc),
                                'pv_consummed': min(pv, load),
                                }
            if load - pv < 0:
                control_dict = {'battery_charge': p_char,
                                'battery_discharge': 0,
                                'grid_import': 0,
                                'grid_export': 0,  # abs(min(load-pv,0)),
                                'genset': 0,
                                # 'pv_consummed': min(pv, load+p_char),
                                'pv_consummed': pv,

                                }

        # you call run passing it your control_dict and it will return the mg_data for the next timestep
        mg_data = microgrid.run(control_dict)
        costs.append(microgrid.get_cost())

    return costs


def run_mpc(microgrid_weak):
    # Run MPC benchmark on microgrid (WEAK)
    microgrid_weak.set_horizon(48)
    microgrid_weak.reset(True)

    mpc_weak = control.ModelPredictiveControl(microgrid_weak)
    sample_weak = dg.return_underlying_data(microgrid_weak).iloc[-4380:-1]
    sample_weak.index = np.arange(len(sample_weak))
    output = mpc_weak.run_mpc_on_sample(sample=sample_weak ,verbose=True)
    mpc_cost_weak=output['cost']['cost']

    # Run MPC benchmark on microgrid (ON-GRID)
    microgrid_on.set_horizon(48)
    microgrid_on.reset(True)

    mpc_on = control.ModelPredictiveControl(microgrid_on)
    sample_on = dg.return_underlying_data(microgrid_on).iloc[-4380:-1]
    sample_on.index = np.arange(len(sample_on))
    output = mpc_on.run_mpc_on_sample(sample=sample_on ,verbose=True)
    mpc_cost_on=output['cost']['cost']

    return mpc_cost_weak
