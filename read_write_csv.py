import pandas as pd
import warnings
from tqdm import TqdmExperimentalWarning
from tqdm import tqdm
import datetime
from traffic.data import airports
import timezonefinder, pytz
from dateutil import parser
import openap
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

tqdm.pandas()

print("Reading challenge_set.csv")
df = pd.read_csv('final_submission_set.csv')

print("1/13 Writing Season, Month, Day of week")
#Season 0: Winter, 1: Spring, 2:Summer, 3:Fall
def get_season(date):
    month = date.month
    day = date.day
    
    if (month == 12 and day >= 21) or month in [1, 2] or (month == 3 and day < 20):
        return 0
    elif (month == 3 and day >= 20) or month in [4, 5] or (month == 6 and day < 21):
        return 1
    elif (month == 6 and day >= 21) or month in [7, 8] or (month == 9 and day < 23):
        return 2
    else:
        return 3

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

df['day_of_week'] = df['date'].dt.dayofweek  # Day of the week
df['month'] = df['date'].dt.month  # Month name
df['season'] = df['date'].apply(get_season)  # Season

print("2/13 Writing Domestic or International")
def flight_type(row):
    if row['country_code_adep'] == row['country_code_ades']:
        return '1'
    else:
        return '0'

# Apply the function to create a new column 'flight_type'
df['domestic'] = df.apply(flight_type, axis=1)

print("3/13 Writing Local Offblock Time and Local Arrival Time")
# Convert 'actual_offblock_time' and 'arrival_time' to datetime with timezone awareness
df['actual_offblock_time'] = pd.to_datetime(df['actual_offblock_time'], utc=True)
df['arrival_time'] = pd.to_datetime(df['arrival_time'], utc=True)

adep_unique_airports = df['adep'].unique()
adep_timezones = {}
ades_unique_airports = df['ades'].unique()
ades_timezones = {}

for airport in tqdm(adep_unique_airports):
    try:
        airport_ap = airports[airport]
        tf = timezonefinder.TimezoneFinder()
        adep_timezones[airport] = tf.certain_timezone_at(lat=airport_ap.latitude, lng=airport_ap.longitude)
    except:
        adep_timezones[airport] = None

df['local_offblock_time'] = df.progress_apply(lambda row: row['actual_offblock_time'].tz_convert(adep_timezones[row['adep']]), axis=1)

for airport in tqdm(ades_unique_airports):
    try:
        airport_ap = airports[airport]
        tf = timezonefinder.TimezoneFinder()
        ades_timezones[airport] = tf.certain_timezone_at(lat=airport_ap.latitude, lng=airport_ap.longitude)
    except:
        ades_timezones[airport] = None

df['local_arrival_time'] = df.progress_apply(lambda row: row['arrival_time'].tz_convert(ades_timezones[row['ades']]), axis=1)

print("4/13 Writing Takeoff and Landing Time of day")
def get_time_of_day(time):
    #local_time = time.tz_convert(None)  # remove the timezone info to compare local hour
    hour = time.hour
    if 4 <= hour < 10:
        return 0
    elif 10 <= hour < 16:
        return 1
    elif 16 <= hour < 20:
        return 2
    else:
        return 3

# Apply the function to determine takeoff and landing time of day
df['takeoff_time_of_day'] = df['local_offblock_time'].apply(get_time_of_day)
df['landing_time_of_day'] = df['local_arrival_time'].apply(get_time_of_day)

print("5/13 Writing Aircraft and Engine Specific parameters")
#Write Aircraft and Engine Specific parameters

def get_aircraft_info(aircraft_type):
    try:
        aircraft = openap.prop.aircraft(aircraft_type)
        # Fetching various data fields if available

        max_takeoff_w = aircraft['mtow']
        max_land_w = aircraft['mlw']
        operating_empty_w = aircraft['oew']

        max_fuel_cap = aircraft['mfc']
        max_pass = aircraft['pax']['max']
        wing_area = aircraft['wing']['area']
        cruise_height = aircraft['cruise']['height']
        mach = aircraft['cruise']['mach']
        engine = aircraft['engine']['default']
        engine_num = aircraft['engine']['number']

        engine_prop = openap.prop.engine(engine)
        max_thrust = engine_prop['max_thrust']
        fuel_lto = engine_prop['fuel_lto']
        ff_to = engine_prop['ff_to']
        bpr = engine_prop['bpr']
        pr = engine_prop['pr']
        cruise_sfc = engine_prop['cruise_sfc']
        cruise_thrust = engine_prop['cruise_thrust']
        
        return pd.Series({
            'max_takeoff_w': max_takeoff_w,
            'max_land_w': max_land_w,
            'operating_empty_w': operating_empty_w,
            'max_fuel_cap': max_fuel_cap,
            'max_pass': max_pass,
            'wing_area': wing_area,
            'cruise_height': cruise_height,
            'mach': mach,
            'engine_num': engine_num,
            'max_thrust': max_thrust,
            'fuel_lto': fuel_lto,
            'ff_to': ff_to,
            'bpr': bpr,
            'pr': pr,
            'cruise_sfc': cruise_sfc,
            'cruise_thrust': cruise_thrust
        })
    except Exception as e:
        # If there is an issue with the aircraft type, return unknown values
        return pd.Series({
            'max_takeoff_w': 0,
            'max_land_w': 0,
            'operating_empty_w': 0,
            'max_fuel_cap': 0,
            'max_pass': 0,
            'wing_area': 0,
            'cruise_height': 0,
            'mach': 0,
            'engine_num': 0,
            'max_thrust': 0,
            'fuel_lto': 0,
            'ff_to': 0,
            'bpr': 0,
            'pr': 0,
            'cruise_sfc': 0,
            'cruise_thrust': 0
        })

# Apply the function to each aircraft_type and append the information to the DataFrame
df[['max_takeoff_w', 'max_land_w', 'operating_empty_w', 'max_fuel_cap', 'max_pass', 'wing_area', 'cruise_height', 'mach', 'engine_num', 'max_thrust', 'fuel_lto', 'ff_to', 'bpr', 'pr', 'cruise_sfc', 'cruise_thrust']] = df['aircraft_type'].progress_apply(get_aircraft_info)


print("6/13 Writing if airline is Large, Medium, Small or Very Small")
# Step 1: Aggregate data to count the number of flights per airline
flight_counts = df['airline'].value_counts().reset_index()
flight_counts.columns = ['airline', 'total_flights']

def classify_airline_size(row):
    if row['total_flights'] >= 30000:  # Threshold for large airlines
        return 3
    elif 10000 <= row['total_flights'] < 30000:  # Threshold for medium airlines
        return 2
    elif 5000 <= row['total_flights'] < 10000:
        return 1
    else:  # Threshold for small airlines
        return 0

flight_counts['airline_size'] = flight_counts.apply(classify_airline_size, axis=1)
df = df.merge(flight_counts[['airline', 'airline_size']], on='airline', how='left')

print("7/13 Encoding country_code_adep country_code_ades")
# Encoding Countries
# Label Encoding
label_encoder = LabelEncoder()
df['country_code_adep_encoded'] = label_encoder.fit_transform(df['country_code_adep'])
df['country_code_ades_encoded'] = label_encoder.fit_transform(df['country_code_ades'])

print("8/13 Writing the size of the airport")
# Calculating the size of the airport
# Step 1: Count number of flights departing from and arriving at each airport
adep_flight_counts = df['adep'].value_counts().reset_index()
ades_flight_counts = df['ades'].value_counts().reset_index()

# Rename columns for clarity
adep_flight_counts.columns = ['airport', 'departure_flights']
ades_flight_counts.columns = ['airport', 'arrival_flights']

# Step 2: Merge departure and arrival flight counts using an outer join
airport_flight_counts = pd.merge(adep_flight_counts, ades_flight_counts, on='airport', how='outer')

# Step 3: Fill missing values with 0 (because not all airports appear in both lists)
airport_flight_counts['departure_flights'] = airport_flight_counts['departure_flights'].fillna(0)
airport_flight_counts['arrival_flights'] = airport_flight_counts['arrival_flights'].fillna(0)

# Step 4: Calculate total flights (departures + arrivals) for each airport
airport_flight_counts['total_flights'] = airport_flight_counts['departure_flights'] + airport_flight_counts['arrival_flights']

# Step 5: Define thresholds for classification
def classify_airport_size(row):
    if row['total_flights'] > 10:  # Example threshold for busy airports
        return '2'
    elif 5 <= row['total_flights'] <= 10:  # Example threshold for medium airports
        return '1'
    else:  # Example threshold for light airports
        return '0'
    
# Step 6: Apply classification based on flight counts
airport_flight_counts['airport_size'] = airport_flight_counts.apply(classify_airport_size, axis=1)

# Step 7: Merge the classification for both 'adep' and 'ades' into the original DataFrame
# Merge the departure airport classification ('adep')
df = df.merge(airport_flight_counts[['airport', 'airport_size']], left_on='adep', right_on='airport', how='left')
df = df.rename(columns={'airport_size': 'adep_size'}).drop(columns=['airport'])

# Merge the arrival airport classification ('ades')
df = df.merge(airport_flight_counts[['airport', 'airport_size']], left_on='ades', right_on='airport', how='left')
df = df.rename(columns={'airport_size': 'ades_size'}).drop(columns=['airport'])

print("9/13 Encoding ADEP, ADES and WTC, airline, aircraft_type")
# Encode ADEP, ADES and WTC, airline, aircraft_type
df['adep_encoded'] = label_encoder.fit_transform(df['adep'])
df['ades_encoded'] = label_encoder.fit_transform(df['ades'])
df['wtc_encoded'] = label_encoder.fit_transform(df['wtc'])
df['airline_encoded'] = label_encoder.fit_transform(df['airline'])
df['aircraft_type_encoded'] = label_encoder.fit_transform(df['aircraft_type'])

print("10/13 Combining route info")
df['route'] = df['adep'] + df['ades']
df['route_encoded'] = label_encoder.fit_transform(df['route'])

print("11/13 Checing if the flight is from or to the US")
us_code = 'US'

# Add a new column for 'flight_direction'
def flight_direction(row):
    if ((row['country_code_adep'] == us_code) | (row['country_code_ades'] == us_code)):
        return 1  # Flight is leaving the US
    else:
        return 0  # Flight is within the US (if both are US), or neither (if neither are US)

# Apply the function to determine the direction of the flight
df['us_flight'] = df.apply(flight_direction, axis=1)

print("12/13 Calculating Capacity Usage Percentage")
max_cap = df.groupby('month')['max_pass'].sum()
total_pass = [31443647, 35451984, 48327468, 64836740, 74935995, 84510587, 94632200, 95723823, 87057609, 81217442, 59001705, 61031507]
max_cap = max_cap * 20
diff = max_cap - total_pass
cap_per = diff/max_cap

df['capacity_usage_percentage'] = df['month'].map(cap_per)

print("13/13 Calculating relations")
df['relation1'] = df['aircraft_type_encoded'] * df['airline_encoded'] * df['route_encoded']
df['fuel_factor'] = (df['flown_distance'] * df['operating_empty_w'])/1000000000

print(df.columns)
print(df)

df.to_csv('test_submission_file.csv', sep=',')
