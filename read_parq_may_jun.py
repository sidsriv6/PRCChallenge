import pandas as pd
import warnings
from tqdm import TqdmExperimentalWarning
from tqdm.notebook import tqdm
import datetime
from traffic.data import airports
import timezonefinder, pytz
from dateutil import parser
import openap
from sklearn.preprocessing import LabelEncoder
from traffic.core import Traffic
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import glob
import datetime

tqdm.pandas()

df = pd.read_csv('test_submission_file.csv')
print(df.columns)

def read_parquet(parquet_file):
        t = (Traffic.from_file(parquet_file).filter().resample('1s').eval())
        return t

# Function to determine the flight phase
def identify_phase(row):
    if row['groundspeed'] > takeoff_groundspeed_min and row['altitude'] <= takeoff_altitude_max and row['vertical_rate'] > 0:
        return 'Takeoff'
    elif row['vertical_rate'] >= climb_vertical_rate_min:
        return 'Climb'
    elif abs(row['vertical_rate']) <= cruise_vertical_rate_max:
        return 'Cruise'
    elif row['vertical_rate'] <= descent_vertical_rate_max:
        return 'Descent'
    else:
        return 'Landing'

#parq_files = glob.glob("./competition-data/*.parquet")

start_date = datetime.date(2022, 5, 1)
end_date = datetime.date(2022, 6, 30)
delta = datetime.timedelta(days=1)

while (start_date <= end_date):
    f_name = str(start_date.year) + '-' + str(start_date.month).zfill(2) + '-' + str(start_date.day).zfill(2) + '.parquet'
    print("Reading ./competition-data/" + f_name)
    tf = read_parquet('./competition-data/' + f_name)

    day_flights = df[pd.to_datetime(df['actual_offblock_time']).dt.date == datetime.date(2022, start_date.month, start_date.day)]
    for item in day_flights['Unnamed: 0']:
        tf_data = tf.data[tf.data['flight_id'] == day_flights['flight_id'][item]]
        loc = day_flights['Unnamed: 0'][item]
        print('Processing: ' + str(loc))
        loc = int(loc)
        arr = day_flights['arrival_time'][item]
        off = day_flights['actual_offblock_time'][item]

        off = datetime.datetime.strptime(off[0:19], '%Y-%m-%d %H:%M:%S')
        off_time = off.time()
        arr = datetime.datetime.strptime(arr[0:19], '%Y-%m-%d %H:%M:%S')
        arr_time = arr.time()

        tf_data['time'] = tf_data['timestamp'].dt.time
        tf_data_truncated = tf_data[tf_data['time'] > off_time]
        tf_data_truncated = tf_data[tf_data['time'] < arr_time]

        tf_data_truncated['track_radians'] = np.deg2rad(tf_data_truncated['track'])
        tf_data_truncated['wind_direction'] = (np.degrees(np.arctan2(tf_data_truncated['v_component_of_wind'], tf_data_truncated['u_component_of_wind'])) + 360) % 360
        tf_data_truncated['wind_direction_radians'] = np.deg2rad(tf_data_truncated['wind_direction'])

        tf_data_truncated['wind_speed'] = np.sqrt(tf_data_truncated['u_component_of_wind']**2 + tf_data_truncated['v_component_of_wind']**2)
        tf_data_truncated['headwind_component'] = tf_data_truncated['wind_speed'] * np.cos(tf_data_truncated['wind_direction_radians'] - tf_data_truncated['track_radians'])

        max = tf_data_truncated['headwind_component'].max()
        min = tf_data_truncated['headwind_component'].min()
        mean = tf_data_truncated['headwind_component'].mean(skipna=True)

        df.at[loc, 'max_headwind_component'] = max
        df.at[loc, 'min_headwind_component'] = min
        df.at[loc, 'mean_headwind_component'] = mean

        # ICAO standard lapse rate in °C per meter (-6.5°C per km)
        lapse_rate = -6.5 / 1000
        sea_level_temp = 288  # Standard temperature at sea level in °C

        # Calculate the standard temperature at each altitude
        tf_data_truncated['standard_temperature'] = sea_level_temp + lapse_rate * tf_data_truncated['altitude']

        # Calculate the temperature difference (ΔT) from the standard temperature
        tf_data_truncated['temperature_difference'] = tf_data_truncated['temperature'] - tf_data_truncated['standard_temperature']

        max = tf_data_truncated['temperature_difference'].max()
        min = tf_data_truncated['temperature_difference'].min()
        mean = tf_data_truncated['temperature_difference'].mean(skipna=True)

        df.at[loc, 'max_temperature_difference'] = max
        df.at[loc, 'min_temperature_difference'] = min
        df.at[loc, 'mean_temperature_difference'] = mean

        takeoff_groundspeed_min = 50       # Minimum groundspeed for takeoff in knots
        takeoff_altitude_max = 3000        # Maximum altitude for takeoff phase in feet
        climb_vertical_rate_min = 500      # Minimum positive vertical rate for climb in feet per minute
        cruise_vertical_rate_max = 100     # Maximum vertical rate for cruise phase
        descent_vertical_rate_max = -500   # Maximum descent rate in feet per minute

        # Apply phase identification to each row
        tf_data_truncated['flight_phase'] = tf_data_truncated.apply(identify_phase, axis=1)

        # Calculate takeoff-specific features
        takeoff_data = tf_data_truncated[tf_data_truncated['flight_phase'] == 'Takeoff']
        average_takeoff_groundspeed = takeoff_data['groundspeed'].mean(skipna=True)
        average_takeoff_climb_rate = takeoff_data['vertical_rate'].mean(skipna=True)

        # Calculate descent-specific features
        descent_data = tf_data_truncated[tf_data_truncated['flight_phase'] == 'Descent']
        average_descent_groundspeed = descent_data['groundspeed'].mean(skipna=True)
        average_descent_climb_rate = descent_data['vertical_rate'].mean(skipna=True)

        # Calculate duration (in seconds) of each phase
        tf_data_truncated['time_diff'] = tf_data_truncated['timestamp'].diff().dt.total_seconds().fillna(0)
        phase_durations = tf_data_truncated.groupby('flight_phase')['time_diff'].sum() 

        # Calculate the percentage of each phase over the entire flight
        total_flight_time = tf_data_truncated['time_diff'].sum()
        phase_percentages = (phase_durations / total_flight_time) * 100

        df.at[loc, 'avg_takeoff_groundspeed'] = average_takeoff_groundspeed
        df.at[loc, 'avg_takeoff_climb_rate'] = average_takeoff_climb_rate
        df.at[loc, 'avg_descent_groundspeed'] = average_descent_groundspeed
        df.at[loc, 'avg_descent_climb_rate'] = average_descent_climb_rate
        try:
            df.at[loc ,'descent_percent'] = phase_percentages['Descent']
        except:
            df.at[loc ,'descent_percent'] = 0
        try:
            df.at[loc ,'takeoff_percent'] = phase_percentages['Takeoff']
        except:
            df.at[loc ,'takeoff_percent'] = 0
        try:
            df.at[loc ,'climb_percent'] = phase_percentages['Climb']
        except:
            df.at[loc ,'climb_percent'] = 0
        try:
            df.at[loc ,'cruise_percent'] = phase_percentages['Cruise']
        except:
            df.at[loc ,'cruise_percent'] = 0
        try:
            df.at[loc ,'landing_percent'] = phase_percentages['Landing']
        except:
            df.at[loc ,'landing_percent'] = 0

        max_alt_climb = tf_data_truncated[tf_data_truncated['flight_phase'] == 'Climb']['altitude'].max() 
        df.at[loc, 'max_alt_climb'] = max_alt_climb

        # Constants for fuel burn rate (simplified constants based on aircraft type, altitude, etc.)
        fuel_burn_climb_factor = 0.8  # Higher burn rate factor for climb phase
        fuel_burn_cruise_factor = 0.4  # Lower burn rate factor for cruise phase
        fuel_burn_descent_factor = 0.2  # Lowest burn rate factor for descent phase

        # Calculate fuel burn rate based on vertical rate, groundspeed, and phase
        def estimate_fuel_burn_rate(row):
            # Basic rate in kg per minute
            base_burn_rate = row['groundspeed'] * 0.1  # Placeholder value

            # Adjust base burn rate by phase
            if row['flight_phase'] == 'Climb':
                return base_burn_rate * (1 + fuel_burn_climb_factor)
            elif row['flight_phase'] == 'Cruise':
                return base_burn_rate * (1 + fuel_burn_cruise_factor)
            elif row['flight_phase'] == 'Descent':
                return base_burn_rate * (1 + fuel_burn_descent_factor)
            else:
                return base_burn_rate

        # Apply the function to each row
        tf_data_truncated['fuel_burn_rate'] = tf_data_truncated.apply(estimate_fuel_burn_rate, axis=1)

        fuel_rate_climb = tf_data_truncated[tf_data_truncated['flight_phase'] == 'Climb']['fuel_burn_rate'].mean(skipna=True)
        fuel_rate_cruise = tf_data_truncated[tf_data_truncated['flight_phase'] == 'Cruise']['fuel_burn_rate'].mean(skipna=True)
        fuel_rate_descent = tf_data_truncated[tf_data_truncated['flight_phase'] == 'Descent']['fuel_burn_rate'].mean(skipna=True)

        df.at[loc, 'fuel_rate_climb'] = fuel_rate_climb
        df.at[loc, 'fuel_rate_cruise'] = fuel_rate_cruise
        df.at[loc, 'fuel_rate_descent'] = fuel_rate_descent
    start_date += delta

df.to_csv('final_file_may_jun.csv', sep=',')
