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
data['Temp_in'] = data[['Temp_in1', 'Temp_in2', 'Temp_in2']].astype(float).mean(1)
data['deltaT'] = data.Temp_in - data.Temp_out
data['Power'] = (data.Heat + data.Cool)/3.6e6
data

model = sm.OLS(data.Power,sm.add_constant(data.deltaT))
results=model.fit()

fig = plt.figure
plt.plot(data.deltaT,results.predict(),'r', linewidth=1, label='Regression Line')
plt.scatter(data.deltaT,data.Power, s=1, label='Observations')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.1)
plt.legend()
plt.savefig('../fig/esobs')
plt.close()

data=data.resample('H').mean()
data=data.dropna()
model = sm.OLS(data.Power,sm.add_constant(data.deltaT))
results=model.fit()

fig = plt.figure
plt.plot(data.deltaT,results.predict(),'r', linewidth=1, label='Regression Line')
plt.scatter(data.deltaT,data.Power, s=5, label='Observations')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.1)
plt.legend()
plt.savefig('../fig/eshourly')
plt.close()

data=data.resample('D').mean()
data=data.dropna()
model = sm.OLS(data.Power,sm.add_constant(data.deltaT))
results=model.fit()

fig = plt.figure
plt.plot(data.deltaT,results.predict(),'r', linewidth=1, label='Regression Line')
plt.scatter(data.deltaT,data.Power, s=5, label='Observations')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.1)
plt.legend()
plt.savefig('../fig/esdaily')
plt.close()

data=data.resample('W').mean()
data=data.dropna()
model = sm.OLS(data.Power,sm.add_constant(data.deltaT))
results=model.fit()

fig = plt.figure
plt.plot(data.deltaT,results.predict(),'r', linewidth=1, label='Regression Line')
plt.scatter(data.deltaT,data.Power, s=10, label='Observations')
plt.xlabel('\u0394T [\u00B0C]')
plt.ylabel('Energy Consumption [kWh]')
plt.ylim(-0.1)
plt.legend()
plt.savefig('../fig/esweekly')
plt.close()


