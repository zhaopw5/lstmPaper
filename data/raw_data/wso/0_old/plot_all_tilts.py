import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
data = pd.read_csv('tilts.csv')

# Convert 'Start Date' including hour info to datetime
data['start_date'] = pd.to_datetime(data['Start Date'].str.replace('h', ''), format="%Y-%m-%d %H")

# Calculate center date between adjacent start dates
data['next_start'] = data['start_date'].shift(-1)
data['date'] = data['start_date'] + (data['next_start'] - data['start_date']) / 2

# For the last row, use the same interval as previous
last_interval = data['date'].iloc[-2] - data['start_date'].iloc[-2]
data.loc[data.index[-1], 'date'] = data['start_date'].iloc[-1] + last_interval

print("Using midpoint dates between adjacent Carrington Rotations")
print(f"Example: Start1 {data['start_date'].iloc[0]} Start2 {data['start_date'].iloc[1]} -> Center {data['date'].iloc[0]}")

# save data:
data.to_csv('tilts_update.csv', index=False)
