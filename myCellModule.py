#Using the price for 2021 to duplicate 15 years of data

import os
import sys
import time as time_measure
import logging
import pandas as pd
import pvlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from timezonefinder import TimezoneFinder
# from blast.models import (  # import some of the blast light models, can import more, and some of the ones you defined
#     Lfp_Gr_250AhPrismatic,
#     Nmc622_Gr_DENSO50Ah_Battery,
#     Nmc111_Gr_Kokam75Ah_Battery
# )
from mylib import Nmc111_Gr_Kokam75Ah_Battery, nmc_SANYOUR18650, nmc_20Ah
 

from blast.utils.functions import derate_profile

# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================
# -File Paths, adjust them to your ows
DATA_DIR = r"D:\SELECT\StoreNow\code\TestingVersion_and_Data-20250717T072506Z-1-001\TestingVersion_and_Data"
LOAD_PROFILE_FILE = os.path.join(DATA_DIR, "LoadProfilesSummary.csv") # 
PRICE_DATA_FILE   = os.path.join(DATA_DIR,"DE_day_ahead_2023.csv") # Using the uploaded file

# --- Simulation Parameters ---
SIMULATION_YEAR   = 2021
ANNUAL_CONSUMPTION_KWH = 150000
TIMEZONE          = "Europe/Berlin"

# --- Location & PV System ---
LATITUDE          = 51.176164
LONGITUDE         = 6.819423
PV_PEAK_POWER_KW  = 150
PV_SURFACE_TILT   = 30
PV_SURFACE_AZIMUTH= 180 # 180=South, 0=North, 90=East, 270=West
PV_LOSSES_PERCENT = 14

# https://www.tesvolt.com/de/produkte/e-serie/ts-i-hv-100-e.html
# One Storage Unit from our Partner "Tesvolt"
# Battery Specifications
battery_model_name = "TS-I-HV 100 E"
battery_type = "NMC"
battery_cost = 50000  # in EUR, just assumed
battery_nominal_capacity_kWh = 288
battery_usable_capacity_kWh = 288*0.95*(1-0.1)  # assume min SOC 10% and max SOC 95%
# ?why not 288*(0.95-0.1)

battery_nominal_power_charge_kW = 97
battery_nominal_power_discharge_kW = 97
battery_efficiency_percent = 97.0
battery_reduction_factor_percent = 0 # factor to consider if usable capacity needs to be reduced
battery_warranty_years = 10
battery_measurements_str = "200x61x99 cm"
battery_weight_kg = 834

# Inverter Specifications, under same link as for battery
inverter_model_name = "Tesvolt IPU"
inverter_cost = 10000  # in EUR, just assumed
inverter_max_pv_power_kW = 168 # maximum 168 kW array
inverter_max_ac_power_kW = 60
inverter_max_bat_charge_power_kW    = 85
inverter_max_bat_discharge_power_kW = 85
inverter_eff_pv_to_ac_percent =  97.0
inverter_eff_pv_to_bat_percent = 97.0
inverter_eff_bat_to_ac_percent = 97.0


# --- Financial & Environmental Parameters ---
CO2_EMISSIONS_GER_G_PER_KWH = 380
CO2_EMISSIONS_CAR_G_PER_KM = 167
DISCOUNT_RATE = 0.02
GRID_SELL_PRICE_EUR_PER_KWH = 0.07    # fixed rate currently in germany for feed in surplus PV, decreases according to law 2% each month
GRID_PRICE_ASSUMED_EUR_PER_KWH = 0.35 # Used for arbitrage calculation baseline
FLAT_TAX_EUR_PER_KWH = 10  # 10 cents/kWh taxing, assumed
VAT_PERCENT = 19.0           # Tax rate in germany

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def get_charging_days(num_days):
    charging_schedule = {
        1: [1], 2: [1, 5], 3: [1, 4, 6], 4: [0, 2, 4, 6],
        5: [0, 1, 2, 4, 5], 6: [0, 1, 2, 3, 4, 5], 7: list(range(7))
    }
    return charging_schedule.get(num_days, [])

η_round = battery_efficiency_percent*pow(inverter_eff_bat_to_ac_percent,2) # estimated round trip efficiency
μ = (1/η_round) - 1

# Schedule for battery behavior('standby', 'charge', 'discharge')
def one_day_states_hourly(p24, spread=μ):
    mins = (p24[:-2] > p24[1:-1]) & (p24[2:] >= p24[1:-1])
    maxs = (p24[:-2] < p24[1:-1]) & (p24[2:] <= p24[1:-1])
    states = np.full(24, "standby", "U8")
    for i_min, i_max in zip(np.where(mins)[0]+1, np.where(maxs)[0]+1):
        if p24[i_max] - p24[i_min] > spread * p24[i_min]:
            states[i_min] = "charge"
            states[i_max] = "discharge"
    return states


def derive_stressors(slice_soc, slice_charge_kWh, slice_discharge_kWh,
                     slice_capacity_kWh, T_ambient_K=298.0):
    delta_t_days = len(slice_soc) * 0.25 / 24
    delta_soc = np.diff(slice_soc, prepend=slice_soc[0])
    delta_efc = 0.5 * np.abs(delta_soc).sum()
    avg_soc = slice_soc.mean()
    avg_dod = np.abs(delta_soc).mean()
    slice_P_chg = slice_charge_kWh / 0.25
    slice_P_dis = slice_discharge_kWh / 0.25
    cvec = np.concatenate([slice_P_chg, slice_P_dis]) / slice_capacity_kWh
    Crate = np.percentile(np.abs(cvec), 95)
    return dict(delta_t_days=delta_t_days, delta_efc=delta_efc,
                soc=avg_soc, dod=avg_dod, Crate=min(Crate, 0.65),
                TdegK=T_ambient_K)

# ==============================================================================
# 3. DATA PREPARATION & TIME SERIES ALIGNMENT
# ==============================================================================
logger.info("Starting data preparation...")
start_time_data_prep = time_measure.time()

# --- 3a. Prepare a full-year 15-minute DatetimeIndex ---
base_index = pd.date_range(
    start=f"{SIMULATION_YEAR}-01-01 00:00",
    end=f"{SIMULATION_YEAR}-12-31 23:45",
    freq="15min",
    tz=TIMEZONE
)

# --- 3b. Load and Scale Household Load Profile ---
try:
    load_df = pd.read_csv(LOAD_PROFILE_FILE, parse_dates=["Date"], index_col="Date")
    load_df.index = pd.to_datetime(load_df.index, utc=True).tz_convert(TIMEZONE)
    load_15min_series = load_df['Load'].resample('15min').ffill().reindex(base_index, method='ffill')
    load_15min_series *= ANNUAL_CONSUMPTION_KWH / load_15min_series.sum()
    logger.info(f"Loaded and scaled household load profile. Annual total: {load_15min_series.sum():.2f} kWh")
except FileNotFoundError:
    logger.error(f"FATAL: Load profile file not found at '{LOAD_PROFILE_FILE}'.")
    sys.exit(1)

# --- 3c. Fetch PV Generation Data ---
# Try to fetch from PVGIS with retry mechanism, fallback to synthetic data if failed
max_retries = 3
retry_delay = 5  # seconds

for attempt in range(max_retries):
    try:
        logger.info(f"Attempting to fetch PV data from PVGIS (attempt {attempt + 1}/{max_retries})...")
        power_dc_df, _ = pvlib.iotools.get_pvgis_hourly(
            latitude=LATITUDE, longitude=LONGITUDE, start=SIMULATION_YEAR, end=SIMULATION_YEAR,
            raddatabase="PVGIS-SARAH3", pvcalculation=True, peakpower=PV_PEAK_POWER_KW,
            pvtechchoice="crystSi", mountingplace="building", loss=PV_LOSSES_PERCENT,
            surface_tilt=PV_SURFACE_TILT, surface_azimuth=PV_SURFACE_AZIMUTH,
            outputformat="json", map_variables=True
        )
        pv_hourly_kW = (power_dc_df["P"] / 1000).tz_convert(TIMEZONE)
        pv_15min_series = pv_hourly_kW.resample("15min").ffill().reindex(base_index, method='ffill') * 0.25
        logger.info(f"Successfully fetched PV generation data. Annual total: {pv_15min_series.sum():.2f} kWh")
        break
    except Exception as e:
        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time_measure.sleep(retry_delay)
        else:
            logger.warning("All PVGIS attempts failed. Generating synthetic PV data...")
            # Generate synthetic PV data based on typical patterns
            hours_in_year = len(base_index)
            day_of_year = np.array([dt.timetuple().tm_yday for dt in base_index])
            hour_of_day = np.array([dt.hour + dt.minute/60 for dt in base_index])
            
            # Solar elevation angle approximation
            solar_elevation = np.maximum(0, np.sin(2 * np.pi * (day_of_year - 81) / 365) * 0.4 + 
                                       np.cos(2 * np.pi * hour_of_day / 24) * 0.6)
            
            # Generate realistic PV profile with seasonal and daily variations
            base_generation = solar_elevation * PV_PEAK_POWER_KW * (1 - PV_LOSSES_PERCENT/100)
            
            # Add some randomness for weather variations
            np.random.seed(42)  # For reproducible results
            weather_factor = np.random.normal(1.0, 0.2, len(base_index))
            weather_factor = np.clip(weather_factor, 0.1, 1.3)  # Reasonable bounds
            
            # Apply weather variations and convert to energy (kWh for 15-min intervals)
            pv_15min_series = pd.Series(base_generation * weather_factor * 0.25, index=base_index)
            pv_15min_series = pv_15min_series.clip(lower=0)  # No negative generation
            
            logger.info(f"Generated synthetic PV data. Annual total: {pv_15min_series.sum():.2f} kWh")

# --- 3d. Load and Process Price Data ---
price_raw_df = pd.read_csv(PRICE_DATA_FILE, usecols=["timestamp", "price"])
price_raw_df["timestamp"] = pd.to_datetime(price_raw_df["timestamp"], utc=True)

# Set timestamp as index. The raw data may have duplicates from DST change,
# so we group by timestamp and average the price.
spot_h_raw = price_raw_df.set_index("timestamp")["price"].groupby(level=0).mean()

# Shift data from its original year (e.g., 2023) to the simulation year.
# A robust way to do this is by reconstructing the datetime objects.
price_year_original = spot_h_raw.index.year.min()
spot_h_shifted = spot_h_raw.copy()
spot_h_shifted.index = spot_h_shifted.index.map(
    lambda dt: dt.replace(year=SIMULATION_YEAR)
)

# Convert to the local timezone for the simulation. This can create duplicate labels
# during the autumn DST "fall back" (e.g., 2 AM happens twice).
spot_h_local = spot_h_shifted.tz_convert(TIMEZONE)

# *** FIX: Resolve DST-induced duplicate timestamps by averaging them. ***
# This is the crucial step to ensure the index is unique before reindexing.
spot_h_unique = spot_h_local.groupby(level=0).mean()

# Create a complete, gapless hourly index for the entire simulation year.
full_hourly_index = pd.date_range(
    start=f"{SIMULATION_YEAR}-01-01 00:00",
    end=f"{SIMULATION_YEAR}-12-31 23:00",
    freq="1H",
    tz=TIMEZONE
)
prices_hourly_complete = spot_h_unique.reindex(full_hourly_index, method='ffill')
price_15min_series = prices_hourly_complete.resample("15min").ffill()
price_15min_series = price_15min_series.reindex(base_index, method='ffill')
price_15min_series_eur = price_15min_series / 1000
price_15min_series_ct = price_15min_series / 10
price_15min_series_ct = (price_15min_series_ct +  FLAT_TAX_EUR_PER_KWH) * (1+VAT_PERCENT/100)

logger.info(f"Successfully loaded and processed price data. Non-null values: {price_15min_series.notna().sum()}/{len(price_15min_series)}")



# --- 3f. Assemble Final Energy DataFrame for one year ---
energy_flow_1Y = pd.DataFrame({
    'solar_generation': pv_15min_series,
    'house_load'      : load_15min_series
})
energy_flow_1Y['total_load'] = energy_flow_1Y['house_load']

# --- 3g. Prepare price and state arrays for the full 15-year simulation ---
price_matrix_h = price_15min_series_ct.resample('1H').first().to_numpy().reshape(-1, 24)
price_1y_ct = np.repeat(price_15min_series_ct.resample('1H').first().to_numpy(), 4)
price_array = np.tile(price_1y_ct, 15)
state_h_1d = [one_day_states_hourly(day) for day in price_matrix_h]
state_h_1y = np.concatenate(state_h_1d)
state_array = np.tile(np.repeat(state_h_1y, 4), 15).astype("U8")

end_time_data_prep = time_measure.time()
logger.info(f"Data preparation complete. Time taken: {end_time_data_prep - start_time_data_prep:.2f} seconds\n")

# ==============================================================================
# 4. SINGLE SYSTEM SIMULATION
# ==============================================================================
logger.info("Starting 15-year simulation for the specified hardware...")
simulation_start_time = time_measure.time()

# --- 4a. Set up simulation parameters from config ---
total_cost = battery_cost + inverter_cost
battery_cap_dod = battery_usable_capacity_kWh * (1.0 + battery_reduction_factor_percent / 100.0)
max_charging_power = min(battery_nominal_power_charge_kW, inverter_max_bat_charge_power_kW)
max_discharging_power = min(battery_nominal_power_discharge_kW, inverter_max_bat_discharge_power_kW)
max_PV_power = inverter_max_pv_power_kW
inverter_eff_PV2AC = inverter_eff_pv_to_ac_percent
inverter_eff_PV2BAT = inverter_eff_pv_to_bat_percent
inverter_eff_BAT2AC = inverter_eff_bat_to_ac_percent
battery_eff_DC2DC = battery_efficiency_percent

# --- 4b. Initialize BLAST-Lite cell and bookkeeping ---
#cell = nmc_SANYOUR18650()   # Here to change the required cell, crucial part!
cell = nmc_20Ah()   # Here to change the required cell, crucial part!

current_capacity = battery_cap_dod
total_years = 15
quarters_per_year = 4
total_quarters = total_years * quarters_per_year
steps_per_year = len(base_index)
steps_per_quarter = steps_per_year // 4    # Every .... month should the capacity be updated? ( here 12/4 = every 3 month)
total_steps = total_years * steps_per_year

# Pre-allocate 15-year result arraysm for all data later extracted from the simulation
battery_charge = np.zeros(total_steps)
battery_discharge_house = np.zeros_like(battery_charge)
battery_discharge_car = np.zeros_like(battery_charge)
battery_soc = np.zeros_like(battery_charge)
grid_electricity_house = np.zeros_like(battery_charge)
grid_electricity_car = np.zeros_like(battery_charge)
grid_to_batt_kWh = np.zeros_like(battery_charge)
direct_pv_house = np.zeros_like(battery_charge)
direct_pv_car = np.zeros_like(battery_charge)
solar_to_grid = np.zeros_like(battery_charge)
clipped_pv = np.zeros_like(battery_charge)

# Build 15-year load/PV array
base_arr = energy_flow_1Y[['solar_generation', 'total_load', 'house_load']].to_numpy(dtype=np.float64)
multi_arr = np.tile(base_arr, (total_years, 1))

cap_history = []
efc_history = []

# --- 4c. QUARTER-BY-QUARTER SIMULATION LOOP for in total 15 years
# Following a quite straight forward charging and discharging logic for self consumption combined with a dynamic charging algorithm
# The algorithm description can be found at this paper: simple operation strategy of battery storage systems
# under dynamic electricity pricing an Italian case study for a medium-scale public facility -> difficult to find free paper use GPT it gave me the link
for q in range(total_quarters):
    start = q * steps_per_quarter
    stop = start + steps_per_quarter
    quarter_arr = multi_arr[start:stop]
    
    rows = steps_per_quarter
    for j in range(rows):
        idx = start + j
        solar_generation = quarter_arr[j, 0]
        house_load = quarter_arr[j, 2]
        #car_load = quarter_arr[j, 3]
        
        # Carry over SOC from previous step
        battery_soc[idx] = battery_soc[idx-1] if idx > 0 else 0

        # Arbitrage based on pre-calculated states
        if state_array[idx] == 'charge' and battery_soc[idx] < current_capacity:
            room = current_capacity - battery_soc[idx]
            gain = min(room, max_charging_power * 0.25)
            # max_charging_power * 0.25 is used to limit the charging power to 25% of the max power
            battery_charge[idx] += gain
            battery_soc[idx] += gain
            # soc is not a percentage?
            grid_to_batt_kWh[idx] = gain
            continue
        elif state_array[idx] == 'discharge' and battery_soc[idx] > 0:
            
            give = min(battery_soc[idx], max_discharging_power * 0.25)
            battery_discharge_house[idx] += give * (inverter_eff_BAT2AC / 100)
            battery_soc[idx] -= give
            continue
        
        # The following is when the battery is in standby mode?

        # Clip PV generation if it exceeds inverter's max PV power
        if solar_generation > (max_PV_power * 0.25):
            clipped_pv[idx] = solar_generation - (max_PV_power * 0.25)
            solar_generation = max_PV_power * 0.25
        
        # Direct PV to loads
        # if car_load and solar_generation:
        #     needed_dc = car_load / (inverter_eff_PV2AC / 100)
        #     served_ac = min(car_load, solar_generation * (inverter_eff_PV2AC / 100))
        #     direct_pv_car[idx] = served_ac
        #     car_load -= served_ac
        #     solar_generation -= served_ac / (inverter_eff_PV2AC / 100)

        if house_load and solar_generation:
            needed_dc = house_load / (inverter_eff_PV2AC / 100)
            served_ac = min(house_load, solar_generation * (inverter_eff_PV2AC / 100))
            direct_pv_house[idx] = served_ac
            house_load -= served_ac
            solar_generation -= served_ac / (inverter_eff_PV2AC / 100)

        # Charge battery from surplus PV
        if solar_generation > 0 and battery_soc[idx] < current_capacity:
            room = current_capacity - battery_soc[idx]
            potential_gain = solar_generation * (inverter_eff_PV2BAT / 100)
            actual_gain = min(room, potential_gain, max_charging_power * 0.25)
            battery_charge[idx] += actual_gain
            battery_soc[idx] += actual_gain
            # the battery_soc here is not a percentage?

            used_dc_for_batt = actual_gain / (inverter_eff_PV2BAT / 100)
            solar_generation -= used_dc_for_batt

        # Feed remaining PV to grid
        if solar_generation > 0:
            solar_to_grid[idx] = solar_generation * (inverter_eff_PV2AC / 100)
        
        # Discharge battery to meet remaining loads
        # if car_load > 0 and battery_soc[idx] > 0:
        #     discharge_ac = min(car_load, battery_soc[idx] * (battery_eff_DC2DC / 100) * (inverter_eff_BAT2AC / 100), max_discharging_power * 0.25)
        #     battery_discharge_car[idx] += discharge_ac
        #     battery_soc[idx] -= discharge_ac / ((inverter_eff_BAT2AC / 100) * (battery_eff_DC2DC / 100))
        #     car_load -= discharge_ac

        if house_load > 0 and battery_soc[idx] > 0:
            #give = min(battery_soc[idx], max_discharging_power * 0.25)
            discharge_ac = min(house_load, battery_soc[idx] * (battery_eff_DC2DC / 100) * (inverter_eff_BAT2AC / 100), max_discharging_power * 0.25)
            battery_discharge_house[idx] += discharge_ac
            battery_soc[idx] -= discharge_ac / ((inverter_eff_BAT2AC / 100) * (battery_eff_DC2DC / 100))
            house_load -= discharge_ac
            
        # Import from grid if load is still unmet
        # if car_load > 0: grid_electricity_car[idx] = car_load
        if house_load > 0: grid_electricity_house[idx] = house_load

    # --- 4d. Update Degradation Model at the end of each quarter ---
    slice_soc = battery_soc[start:stop] / current_capacity
    slice_chg = battery_charge[start:stop]
    slice_dis = battery_discharge_house[start:stop] #+ battery_discharge_car[start:stop]
    
    stress = derive_stressors(slice_soc, slice_chg, slice_dis, current_capacity, T_ambient_K=298.0)
    
    t_secs = np.arange(len(slice_soc)) * 900
    T_celsius = np.full_like(slice_soc, 25) #25 Temperature
    # which type of temperature will be used in the final simulation, cell-temperature or ambient temperature?
    
    # Update the cell with time， SoC and temperature
    cell.update_battery_state(t_secs, slice_soc, T_celsius)
    q_remaining = cell.outputs['q'][-1]
    current_capacity = battery_cap_dod * q_remaining
    cap_history.append(q_remaining)
    efc_history.append(cell.stressors['efc'][-1])

logger.info(f"Simulation finished. Time taken: {time_measure.time() - simulation_start_time:.2f} seconds")

# ==============================================================================
# 5. ANALYSIS AND KPI CALCULATION
# ==============================================================================
# --- 5a. Create the full 15-year results DataFrame ---
multi_index = pd.date_range(start=base_index.min(), periods=total_steps, freq='15min', tz=TIMEZONE)

energy_flow_15Y = pd.DataFrame(index=multi_index)
energy_flow_15Y['solar_generation'] = multi_arr[:, 0]
energy_flow_15Y['house_load'] = multi_arr[:, 2]
#energy_flow_15Y['car_load'] = multi_arr[:, 3]
energy_flow_15Y['total_load'] = multi_arr[:, 1]
energy_flow_15Y['clipping'] = clipped_pv
energy_flow_15Y['battery_charge'] = battery_charge
energy_flow_15Y['battery_discharge_house'] = battery_discharge_house
energy_flow_15Y['battery_discharge_car'] = battery_discharge_car
energy_flow_15Y['battery_soc'] = battery_soc
energy_flow_15Y['direct_pv_house'] = direct_pv_house
energy_flow_15Y['direct_pv_car'] = direct_pv_car
energy_flow_15Y['grid_electricity_house'] = grid_electricity_house
energy_flow_15Y['grid_electricity_car'] = grid_electricity_car
energy_flow_15Y['solar_to_grid'] = solar_to_grid
energy_flow_15Y['grid_to_batt_kWh'] = grid_to_batt_kWh
energy_flow_15Y["SOC_profile"] = battery_soc / battery_cap_dod
energy_flow_15Y["grid_electricity_tot"] = energy_flow_15Y['grid_electricity_house'] + energy_flow_15Y['grid_electricity_car']
energy_flow_15Y["direct_cons_tot"] = energy_flow_15Y['direct_pv_house'] + energy_flow_15Y['direct_pv_car']
energy_flow_15Y['battery_discharge_tot'] = energy_flow_15Y['battery_discharge_house'] + energy_flow_15Y['battery_discharge_car']
energy_flow_15Y["total_cons_ren_house"] = energy_flow_15Y['battery_discharge_house'] + energy_flow_15Y['direct_pv_house']
energy_flow_15Y["total_cons_ren_car"] = energy_flow_15Y['battery_discharge_car'] + energy_flow_15Y['direct_pv_car']
energy_flow_15Y["total_cons_ren_tot"] = energy_flow_15Y['battery_discharge_tot'] + energy_flow_15Y["direct_cons_tot"]

# --- 5b. Calculate Financial KPIs ---
energy_flow_15Y['Price_ct'] = price_array
energy_flow_15Y['Price_eur'] = energy_flow_15Y['Price_ct'] / 100.0

energy_flow_15Y['Battery Savings Tot'] = energy_flow_15Y['battery_discharge_tot'] * energy_flow_15Y['Price_eur']
energy_flow_15Y['Solar Savings Tot'] = energy_flow_15Y["direct_cons_tot"] * energy_flow_15Y['Price_eur']
energy_flow_15Y['Batt_arbitrage_€'] = energy_flow_15Y['grid_to_batt_kWh'] * (GRID_PRICE_ASSUMED_EUR_PER_KWH - energy_flow_15Y['Price_eur'])
energy_flow_15Y['Grid Cost'] = energy_flow_15Y['grid_electricity_tot'] * energy_flow_15Y['Price_eur']
energy_flow_15Y['Grid Revenue'] = energy_flow_15Y['solar_to_grid'] * GRID_SELL_PRICE_EUR_PER_KWH

# Yearly financial summary
yearly = energy_flow_15Y.resample('YE').sum()
annual_benefits = yearly['Battery Savings Tot'] + yearly['Solar Savings Tot'] + yearly['Batt_arbitrage_€'] + yearly['Grid Revenue'] - yearly['Grid Cost']

# NPV and DPBT calculation
capex = total_cost
years = np.arange(1, len(annual_benefits) + 1)
discount_facs = 1.0 / (1.0 + DISCOUNT_RATE) ** years
dcf = np.concatenate(([-capex], annual_benefits.values * discount_facs))
npv = dcf.sum()
cum_dcf = np.cumsum(dcf)
idx_pos = np.where(cum_dcf >= 0)[0]
dpbt = ((idx_pos[0] - 1) + (-cum_dcf[idx_pos[0]-1] / (cum_dcf[idx_pos[0]] - cum_dcf[idx_pos[0]-1]))) if len(idx_pos) > 0 else float('inf')


# --- 5c. Calculate Performance & Environmental KPIs ---
Total_Consumption_tot = energy_flow_15Y['house_load'].sum()
Total_Consumption_ren_tot = energy_flow_15Y["total_cons_ren_tot"].sum()
self_sufficiency_tot = (Total_Consumption_ren_tot / Total_Consumption_tot * 100) if Total_Consumption_tot > 0 else 0

Tot_Co2_saved_kg = Total_Consumption_ren_tot * CO2_EMISSIONS_GER_G_PER_KWH / 1000

# ==============================================================================
# 6. RESULTS AND VISUALIZATION
# ==============================================================================
print("\n--- SIMULATION RESULTS ---")
print(f"System Cost (CAPEX): {capex:.2f} €")
print(f"Net Present Value (NPV) over 15 years: {npv:.2f} €")
print(f"Discounted Payback Time (DPBT): {dpbt:.2f} years")
print(f"Total Self-Sufficiency: {self_sufficiency_tot:.1f} %")
print(f"Total CO2 Saved: {Tot_Co2_saved_kg:.0f} kg")

# --- Degradation metrics ---
remaining_cap_percent = round(cap_history[-1] * 100, 2)
YoY_Degradation_percent = round(((cap_history[0] - cap_history[-1]) / total_years) * 100, 2)
Full_Cycles = round(efc_history[-1], 0)
print(f"  \nBattery Health after 15 years:")
print(f"  Remaining Capacity: {remaining_cap_percent}%")
print(f"  Average Yearly Degradation: {YoY_Degradation_percent}%")
print(f"  Total Equivalent Full Cycles: {Full_Cycles}")

# --- Plot Degradation ---
quarters = np.arange(1, len(cap_history) + 1)
years_frac = quarters / 4.0
plt.figure(figsize=(8, 4))
plt.plot(years_frac, [c*100 for c in cap_history], marker='o', linestyle='-')
plt.xlabel('Years since commissioning')
plt.ylabel('Remaining capacity [%]')
plt.title('Quarterly Capacity Degradation (15 Years)')
plt.grid(True, alpha=0.3)
plt.ylim(bottom=30, top=101)
plt.tight_layout()
plt.show()

# ==============================
# 6. RESULTS AND VISUALIZATION
# ==============================

def plot_weekly_energy_and_price(start_date: str, label: str):
    """
    Creates a two-part plot for a 7-day period showing:
    1. Top plot: Key energy flows (SOC, Load, PV, Grid Charging) with a separate axis for Battery SOC.
    2. Bottom plot: Day-ahead electricity price.
    """
    t0 = pd.Timestamp(start_date, tz=TIMEZONE)
    t1 = t0 + pd.Timedelta(days=7)
    span = energy_flow_15Y.loc[t0:t1].copy()

    fig, (ax_energy, ax_price) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    
    # --- Top Plot: Energy Flows ---
    ax_energy.set_title(f'Energy Flows – Sample Week in {label}', fontsize=14)
    
    # Plotting major flows as lines
    ax_energy.plot(span.index, span['total_load'], label='Total Load [kWh]', color='black', lw=2.5)
    ax_energy.plot(span.index, span['solar_generation'], label='PV Generation [kWh]', color='orange', lw=2)
    ax_energy.bar(span.index, span['grid_to_batt_kWh'], width=0.01, label='Grid Charging [kWh]', color='crimson', alpha=0.8)
    
    ax_energy.set_ylabel('Energy in Interval [kWh]', fontsize=12)
    ax_energy.grid(True, linestyle='--', alpha=0.5)
    ax_energy.set_ylim(bottom=0)

    # Create a secondary Y-axis for the Battery SOC state
    ax_soc = ax_energy.twinx()
    ax_soc.plot(span.index, span['battery_soc'], label='Battery SOC [kWh]', color='purple', linestyle='--')
    ax_soc.set_ylabel('State of Charge [kWh]', fontsize=12, color='purple')
    ax_soc.tick_params(axis='y', labelcolor='purple')
    ax_soc.set_ylim(bottom=0, top=battery_nominal_capacity_kWh)

    # Combine legends from both axes for clarity
    lines, labels = ax_energy.get_legend_handles_labels()
    lines2, labels2 = ax_soc.get_legend_handles_labels()
    ax_soc.legend(lines + lines2, labels + labels2, loc='upper left')

    # --- Bottom Plot: Price ---
    ax_price.plot(span.index, span['Price_eur'], label='Day-Ahead Price [€/kWh]', color='teal')
    ax_price.axhline(0, color='black', lw=0.5, linestyle='--')
    ax_price.set_ylabel('Price [€/kWh]', fontsize=12)
    ax_price.grid(True, linestyle='--', alpha=0.5)
    ax_price.legend(loc='upper left')
    
    ax_price.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d-%b'))
    plt.xlabel(f'Date in {SIMULATION_YEAR}', fontsize=12)
    
    plt.tight_layout()
    plt.show()


def plot_monthly_value_contribution(simulation_df: pd.DataFrame):
    """
    Creates a stacked bar chart showing the monthly financial value generated by the system,
    broken down by its source (self-consumption, arbitrage, and grid export revenue).
    """
    # Isolate the first year of the simulation for the plot
    df = simulation_df.loc[f'{SIMULATION_YEAR}'].copy()
    
    # Calculate the value of energy from different sources
    df['Self-Consumption Value'] = (df['direct_cons_tot'] + df['battery_discharge_tot']) * GRID_PRICE_ASSUMED_EUR_PER_KWH
    df['Arbitrage Value'] = df['Batt_arbitrage_€']
    df['Grid Export Revenue'] = df['solar_to_grid'] * GRID_SELL_PRICE_EUR_PER_KWH

    # Resample all value streams to monthly sums
    monthly_values = df[[
        'Self-Consumption Value',
        'Arbitrage Value',
        'Grid Export Revenue'
    ]].resample('ME').sum()

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 7))
    monthly_values.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=['#2ca02c', '#9467bd', '#1f77b4'], # Green, Purple, Blue
        edgecolor='black'
    )

    ax.set_title(f'Monthly Financial Value Contribution in {SIMULATION_YEAR}', fontsize=14)
    ax.set_ylabel('Value [€]', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add total value labels on top of each bar
    for i, total in enumerate(monthly_values.sum(axis=1)):
        ax.text(i, total, f"€{total:.0f}", ha='center', va='bottom', weight='bold')

    ax.set_xticklabels([d.strftime('%b') for d in monthly_values.index], rotation=0)
    plt.tight_layout()
    plt.show()


# --- Example Calls for the new plotting functions ---
logger.info("Generating visualization plots...")

# Plot sample weeks for energy and price dynamics
plot_weekly_energy_and_price(f'{SIMULATION_YEAR}-07-10', 'July')
plot_weekly_energy_and_price(f'{SIMULATION_YEAR}-01-10', 'January')

# Plot the monthly savings breakdown for the first year of the simulation
plot_monthly_value_contribution(energy_flow_15Y)

