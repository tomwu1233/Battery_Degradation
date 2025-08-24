
import numpy as np  
from mylib.degradation_model import BatteryDegradationModel


class NMC111_SAMSUNG94AH_SR(BatteryDegradationModel):
    

    def __init__(self, degradation_scalar: float = 1, label: str = "NMC111-SAMSUNG94AH_Symbolic_Regression"):
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
            'dod': [0.6, 0.75],
            'soc': [0.05, 0.9],
            'max_Crate': 2,
            'min_Crate': 0.75
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
            
            # Cycle fade parameters, my fit model parameters, Symbolic Regression
            
            'a1':  2.1091e-05,
            'a2': -3.60868e-05,
            'a3':  2.70924e-05,
            'a4':  2.93206e-05,
            'a5': -2.1673e-05,
            'a6': -3.71462e-07,
            'a7': -2.58448e-06,
            'pcyc': 1.17305
            

            # 'b0':  0.00017827,
            # 'b1': -7.33617e-06,   # T
            # 'b2': -7.19585e-07,   # D
            # 'b3':  3.4169e-05,    # C
            # 'b4': -1.09412e-05,   # C^2
            # 'b5':  1.14116e-05,   # T*C^2
            # 'b6':  8.17324e-06,   # C^2.5
            # 'b7': -7.55551e-06,   # C^3.0
            # 'pcyc': 1.17305       # 幂指数


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
        TdegKN = TdegK / (273.15 + 35)

        
        # Grab parameters
        p = self._params_life

        # Calculate the degradation coefficients
        kcal = (np.abs(p['p1'])
            * np.exp(p['p2']/TdegK)
            * np.exp(p['p3']*Ua/TdegK)
        )
        # kcyc = ((p['p4'] + p['p5']*dod + p['p6']*Crate)
        #       * (np.exp(p['p7']/TdegK) + np.exp(-p['p8']/TdegK)))

        # This formula is based on the symbolic regression model
        kcyc = np.abs(
            p['a1'] * TdegKN
            + p['a2'] * dod
            + p['a3'] * Crate
            + p['a4'] * TdegKN * Crate
            + p['a5'] * (TdegKN ** 2) * Crate
            + p['a6'] * (Crate ** 2.5)
            + p['a7'] * (Crate ** 3.0)
            # p['b0']
            # + p['b1'] * TdegKN
            # + p['b2'] * dod
            # + p['b3'] * Crate
            # + p['b4'] * (Crate ** 2)
            # + p['b5'] * TdegKN * (Crate ** 2)
            # + p['b6'] * (Crate ** 2.5)
            # + p['b7'] * (Crate ** 3.0)
        )
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
    
    #learn from nca_grsi_SonyMurata2p5Ah_2023.py
    #20250821
    def _extract_stressors(self, t_secs, soc, T_celsius):
        # extract the usual stressors
        stressors = BatteryDegradationModel._extract_stressors(self, t_secs, soc, T_celsius)
        # model specific stressors: soc_low, soc_high, Cchg, Cdis
        soc_low = np.min(soc)
        soc_high = np.max(soc)
        t_days = t_secs / (24*60*60)
        delta_t_days = t_days[-1] - t_days[0]
        dt = np.diff(t_secs)
        abs_instantaneous_crate = np.abs(np.diff(soc)/np.diff(t_secs/(60*60))) # get instantaneous C-rates
        abs_instantaneous_crate[abs_instantaneous_crate < 1e-2] = 0 # get rid of extremely small values (storage) before calculating mean
        Crate = np.trapz(abs_instantaneous_crate, t_days[1:]) / delta_t_days
        # Check storage condition, which will give nan Crate:
        if np.isnan(Crate):
            Cdis = 0
            Cchg = 0
        else:
            instantaneous_crate = np.diff(soc)/(dt/3600)
            instantaneous_crate[np.abs(instantaneous_crate) < 1e-2] = 0 # threshold tiny values to zero to prevent weirdness
            mask_cchg = instantaneous_crate > 0
            mask_cdis = instantaneous_crate < 0
            instantaneous_cchg = instantaneous_crate[mask_cchg]
            dt_charging = dt[mask_cchg]
            t_secs_charging = np.cumsum(dt_charging)
            if len(instantaneous_cchg) > 1:
                Cchg = np.trapz(instantaneous_cchg, t_secs_charging) / (t_secs_charging[-1] - t_secs_charging[0])
            elif len(instantaneous_cchg) == 1:
                Cchg = instantaneous_cchg[0]
            else: # half cycle with no charge segment
                Cchg = 0
            instantaneous_cdis = instantaneous_crate[mask_cdis]
            dt_discharging = dt[mask_cdis]
            t_secs_discharging = np.cumsum(dt_discharging)
            if len(instantaneous_cdis) > 1:
                Cdis = np.trapz(np.abs(instantaneous_cdis), t_secs_discharging) / (t_secs_discharging[-1] - t_secs_discharging[0])
            elif len(instantaneous_cdis) == 1:
                Cdis = np.abs(instantaneous_cdis)
                Cdis = Cdis[0]
            else: # half cycle with no discharge segment
                Cdis = 0
        # Similar to EFC, C-rate is in units of Amps/Amp*hours nominal, not percent SOC per hour, so rescale by SOH to correct units
        Cchg = Cchg * self.outputs["q"][-1]
        Cdis = Cdis * self.outputs["q"][-1]

        stressors['soc_low'] = soc_low
        stressors['soc_high'] = soc_high
        stressors['Cchg'] = Cchg
        stressors['Cdis'] = Cdis
        return stressors
