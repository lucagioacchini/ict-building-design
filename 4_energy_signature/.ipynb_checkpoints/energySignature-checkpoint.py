import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
from datetime import datetime
from datetime import timedelta  

cols = [
    'BLOCK1:ZONE3:Zone Operative Temperature [C](TimeStep)',
    'BLOCK1:ZONE2:Zone Operative Temperature [C](TimeStep)',
    'BLOCK1:ZONE1:Zone Operative Temperature [C](TimeStep)',
    'DistrictCooling:Facility [J](TimeStep)',
    'DistrictHeating:Facility [J](TimeStep)',
    'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)',
    'Environment:Site Outdoor Air Barometric Pressure [Pa](TimeStep)',
    'Environment:Site Wind Speed [m/s](TimeStep)',
    'BLOCK1:ZONE1:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)',
    'BLOCK1:ZONE2:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)',
    'BLOCK1:ZONE3:Zone Air Relative Humidity [%](TimeStep)',
    'BLOCK1:ZONE2:Zone Air Relative Humidity [%](TimeStep)',
    'BLOCK1:ZONE1:Zone Air Relative Humidity [%](TimeStep)',
]
meas = {
    'BLOCK1:ZONE3:Zone Operative Temperature [C](TimeStep)':'Temp_in3',
    'BLOCK1:ZONE2:Zone Operative Temperature [C](TimeStep)':'Temp_in2',
    'BLOCK1:ZONE1:Zone Operative Temperature [C](TimeStep)':'Temp_in1',
    'DistrictCooling:Facility [J](TimeStep)':'Cool',
    'DistrictHeating:Facility [J](TimeStep)':'Heat',
    'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)':'Temp_out',
    'Environment:Site Outdoor Air Barometric Pressure [Pa](TimeStep)':'Bar_out',
    'Environment:Site Wind Speed [m/s](TimeStep)':'Wind',
    'BLOCK1:ZONE1:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)':'SolarRad1',
    'BLOCK1:ZONE2:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)':'SolarRad2',
    'BLOCK1:ZONE3:Zone Air Relative Humidity [%](TimeStep)':'Humidity3',
    'BLOCK1:ZONE2:Zone Air Relative Humidity [%](TimeStep)':'Humidity2',
    'BLOCK1:ZONE1:Zone Air Relative Humidity [%](TimeStep)':'Humidity1'
}


df=pd.read_csv('../data/ensig/epraw.csv',sep=',',decimal=',',index_col=0, low_memory=False)
# Convert the string index to datetime index
data = pd.DataFrame(df, columns=cols).astype(float)
data = data.rename(columns = meas)
tmstmp = []
for i in range(len(data.index)):
    tt = df.index[i]
    days, hours = tt.split('  ')
    tt = f'{days}/2017{hours}'
    tt = tt.replace(' ', '')
    if '201724:' in tt:
        tt=tt.replace('24:', '00:')
        timestamp = datetime.strptime(tt, "%m/%d/%Y%H:%M:%S")
        timestamp += timedelta(days=1)
    else:
        timestamp = datetime.strptime(tt, "%m/%d/%Y%H:%M:%S")
    tmstmp.append(timestamp)
data.index = pd.to_datetime(tmstmp)
# Determine the internal temperature
data['Temp_in'] = data[['Temp_in1', 'Temp_in2', 'Temp_in3']].astype(float).mean(1)
# Determine the internal/external temperature difference
data['deltaT'] = data.Temp_in - data.Temp_out
data['Power'] = (data.Heat + data.Cool)/3.6e6
o_data = data

#==========================================
# Original Observation
#==========================================
# Extract the cooling and heating dataframes
cool_d = data.where(data['Cool']!=0.0).dropna()
heat_d = data.where(data['Heat']!=0.0).dropna()
# Fit the regression models
model_h = sm.OLS(heat_d.Heat/3.6e6,sm.add_constant(heat_d.deltaT))
model_c = sm.OLS(cool_d.Cool/3.6e6,sm.add_constant(cool_d.deltaT))
results_h = model_h.fit()
results_c = model_c.fit()
# Plot the results
fig = plt.figure()
plt.plot(heat_d.deltaT,results_h.predict(),'r', linewidth=1, label='Heating Regression Line')
plt.plot(cool_d.deltaT,results_c.predict(),'k', linewidth=1, label='Cooling Regression Line')
plt.scatter(heat_d.deltaT,heat_d.Heat/3.6e6, s=2, label='Observations')
plt.scatter(cool_d.deltaT,cool_d.Cool/3.6e6, s=2, color='C0')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.01,1.27)
plt.legend()
#plt.savefig('../fig/esobs.png', transparent=True)
#plt.close()

results_c.params

#==========================================
# Hourly Resempled Data
#==========================================
# Extract the resampled cooling and heating dataframes
data = o_data
data=data.resample('H').mean()
data=data.dropna()
cool_d = data.where(data['Cool']!=0.0).dropna()
heat_d = data.where(data['Heat']!=0.0).dropna()
# Fit the regression models
model_h = sm.OLS(heat_d.Heat/3.6e6,sm.add_constant(heat_d.deltaT))
model_c = sm.OLS(cool_d.Cool/3.6e6,sm.add_constant(cool_d.deltaT))
results_h = model_h.fit()
results_c = model_c.fit()
# Plot the results
fig = plt.figure()
plt.plot(heat_d.deltaT,results_h.predict(),'r', linewidth=1, label='Heating Regression Line')
plt.plot(cool_d.deltaT,results_c.predict(),'k', linewidth=1, label='Cooling Regression Line')
plt.scatter(heat_d.deltaT,heat_d.Heat/3.6e6, s=3, label='Observations')
plt.scatter(cool_d.deltaT,cool_d.Cool/3.6e6, s=3, color='C0')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.01,1.27)
plt.legend()
#plt.savefig('../fig/eshourly', transparent=True)
#plt.close()

results_c.params

#==========================================
# Daily Resempled Data
#==========================================
# Extract the resampled cooling and heating dataframes
data = o_data
data=data.resample('D').mean()
data=data.dropna()
cool_d = data.where(data['Cool']!=0.0).dropna()
heat_d = data.where(data['Heat']!=0.0).dropna()
# Fit the regression models
model_h = sm.OLS(heat_d.Heat/3.6e6,sm.add_constant(heat_d.deltaT))
model_c = sm.OLS(cool_d.Cool/3.6e6,sm.add_constant(cool_d.deltaT))
results_h = model_h.fit()
results_c = model_c.fit()
# Plot the results
fig = plt.figure()
plt.plot(heat_d.deltaT,results_h.predict(),'r', linewidth=1, label='Heating Regression Line')
plt.plot(cool_d.deltaT,results_c.predict(),'k', linewidth=1, label='Cooling Regression Line')
plt.scatter(heat_d.deltaT,heat_d.Heat/3.6e6, s=3, label='Observations')
plt.scatter(cool_d.deltaT,cool_d.Cool/3.6e6, s=3, color='C0')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.01,.53)
plt.legend()
#plt.savefig('../fig/esdaily', transparent=True)
#plt.close()

results_c.params

#==========================================
# Weekly Resempled Data
#==========================================
# Extract the resampled cooling and heating dataframes
data = o_data
data=data.resample('W').mean()
data=data.dropna()
cool_d = data.where(data['Cool']!=0.0).dropna()
heat_d = data.where(data['Heat']!=0.0).dropna()
# Fit the regression models
model_h = sm.OLS(heat_d.Heat/3.6e6,sm.add_constant(heat_d.deltaT))
model_c = sm.OLS(cool_d.Cool/3.6e6,sm.add_constant(cool_d.deltaT))
results_h = model_h.fit()
results_c = model_c.fit()
# Plot the results
fig = plt.figure()
plt.plot(heat_d.deltaT,results_h.predict(),'r', linewidth=1, label='Heating Regression Line')
plt.plot(cool_d.deltaT,results_c.predict(),'k', linewidth=1, label='Cooling Regression Line')
plt.scatter(heat_d.deltaT,heat_d.Heat/3.6e6, s=3, label='Observations')
plt.scatter(cool_d.deltaT,cool_d.Cool/3.6e6, s=3, color='C0')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.01,.5)
plt.legend()
#plt.savefig('../fig/esweekly', transparent=True)
#plt.close()







results_c.params


