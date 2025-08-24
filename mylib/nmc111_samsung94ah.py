import numpy as np  
from mylib.degradation_model import BatteryDegradationModel


class NMC111_SAMSUNG94AH(BatteryDegradationModel):

    def __init__(self, degradation_scalar: float = 1, label: str = "NMC111-SAMSUNG94AH"):
        # States: Internal states of the battery model
        self.states = {
            'qLoss_t': np.array([0]),
            'qLoss_EFC': np.array([0]),
        }

        # Outputs: Battery properties derived from state values
        self.outputs = {
            'q': np.array([1]),
            'q_t': np.array([1]),
            'q_EFC': np.array([1]),
        }

        # Stressors: History of stressors on the battery
        self.stressors = {
            'delta_t_days': np.array([np.nan]), 
            't_days': np.array([0]),
            'delta_efc': np.array([np.nan]), 
            'efc': np.array([0]),
            'TdegK': np.array([np.nan]),
            'soc': np.array([np.nan]), 
            'Ua': np.array([np.nan]), 
            'dod': np.array([np.nan]), 
            'Crate': np.array([np.nan]),
        }

        # Rates: History of stressor-dependent degradation rates
        self.rates = {
            'kcal': np.array([np.nan]),
            'kcyc': np.array([np.nan]),
        }

        # Expermental range: details on the range of experimental conditions, i.e.,
        # the range we expect the model to be valid in
        self.experimental_range = {
            'cycling_temperature': [10, 45],
            'dod': [0.8, 1],
            'soc': [0, 1],
            'max_rate_charge': 0.65,
            'max_rate_discharge': 1,
        }

        # Degradation scalar - scales all state changes by a coefficient
        self._degradation_scalar = degradation_scalar
        # Label for plotting
        self._label = label

    # Nominal capacity
    @property
    def cap(self):
        return 94

    # Define life model parameters
    @property
    def _params_life(self):
        return {
            # Capacity fade parameters

            # calendar aging parameters, NREL LFP250
            'p1': 8.37e+04,
            'p2': -5.21e+03,
            'p3': -3.56e+03,
            'pcal': 0.526,
            
            # Cycle fade parameters, my fit model parameters
            'p4': 3.7425e-06,  # Cycle aging coefficient [1/(K²·Ah)]
            'p5': 1.7872e-07,  # Cycle aging coefficient [1/(K·Ah)]
            'p6': 1.0011e-04,  # Cycle aging coefficient    [1/Ah]  
            'p7': 5.5463e-01,  # C-rate temperature coefficient [K⁻¹]
            'p8': 8.4698e+10,  # C-rate temperature coefficient [K⁻¹]
            'pcyc': 9.7350e-01,  # C-rate temperature coefficient [K⁻¹]


        }
    
    def update_rates(self, stressors):
        # Calculate and update battery degradation rates based on stressor values
        # Inputs:
        #   stressors (dict): output from extract_stressors

        # Unpack stressors
        t_secs = stressors["t_secs"]
        delta_t_secs = t_secs[-1] - t_secs[0]
        TdegK = stressors["TdegK"]
        soc = stressors["soc"]
        Ua = stressors["Ua"]
        dod = stressors["dod"]
        Crate = stressors["Crate"]
        
        # Grab parameters
        p = self._params_life

        # Calculate the degradation coefficients
        kcal = (np.abs(p['p1'])
            * np.exp(p['p2']/TdegK)
            * np.exp(p['p3']*Ua/TdegK)
        )
        kcyc = ((p['p4'] + p['p5']*dod + p['p6']*Crate)
              * (np.exp(p['p7']/TdegK) + np.exp(-p['p8']/TdegK)))
        
        # Calculate time based average of each rate
        kcal = np.trapz(kcal, x=t_secs) / delta_t_secs
        kcyc = np.trapz(kcyc, x=t_secs) / delta_t_secs

        # Store rates
        rates = np.array([kcal, kcyc])
        for k, v in zip(self.rates.keys(), rates):
            self.rates[k] = np.append(self.rates[k], v)
    
    def update_states(self, stressors):
        # Update the battery states, based both on the degradation state as well as the battery performance
        # at the ambient temperature, T_celsius
        # Inputs:
            #   stressors (dict): output from extract_stressors
            
        # Unpack stressors
        delta_t_days = stressors["delta_t_days"]
        delta_efc = stressors["delta_efc"]
        
        # Grab parameters
        p = self._params_life

        # Grab rates, only keep most recent value
        r = self.rates.copy()
        for k, v in zip(r.keys(), r.values()):
            r[k] = v[-1]

        # Calculate incremental state changes
        states = self.states
        # Capacity
        dq_t = self._degradation_scalar * self._update_power_state(states['qLoss_t'][-1], delta_t_days, r['kcal'], p['pcal'])
        dq_EFC = self._degradation_scalar * self._update_power_state(states['qLoss_EFC'][-1], delta_efc, r['kcyc'], p['pcyc'])

        # Accumulate and store states
        dx = np.array([dq_t, dq_EFC])
        for k, v in zip(states.keys(), dx):
            x = self.states[k][-1] + v
            self.states[k] = np.append(self.states[k], x)
    
    def update_outputs(self, stressors):
        # Calculate outputs, based on current battery state
        states = self.states

        # Capacity
        q_t = 1 - states['qLoss_t'][-1]
        q_EFC = 1 - states['qLoss_EFC'][-1]
        q = 1 - states['qLoss_t'][-1] - states['qLoss_EFC'][-1]

        # Assemble output
        out = np.array([q, q_t, q_EFC])
        # Store results
        for k, v in zip(list(self.outputs.keys()), out):
            self.outputs[k] = np.append(self.outputs[k], v)
