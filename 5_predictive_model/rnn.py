import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import timedelta  
from keras.utils import plot_model

df=pd.read_csv('../data/ensig/epraw.csv',sep=',',decimal=',',index_col=0, low_memory=False)

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

def scaling(X, y):
    global scaler
    
    data = np.concatenate((X,y),axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)

    y = data[data.shape[1]-1].to_numpy()
    X = data[range(data.shape[1]-1)].to_numpy()
    X = X.reshape(X.shape[0], 1, X.shape[1])

    return X, y

def descaling(x_test, y_test, results):
    # Descaling
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[2])
    y_test = y_test.reshape(len(y_test), 1)
    results = results.reshape(len(results), 1)

    dset1 = np.concatenate((x_test, y_test),axis=1)
    dset2 = np.concatenate((x_test, results), axis=1)

    dset1 = scaler.inverse_transform(dset1)
    dset2 = scaler.inverse_transform(dset2)


    y = dset1[:,dset2.shape[1]-1]
    y_hat = dset2[:,dset2.shape[1]-1]

    return y, y_hat

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
        #timestamp += 86400
    else:
        timestamp = datetime.strptime(tt, "%m/%d/%Y%H:%M:%S")
    tmstmp.append(timestamp)
data.index = pd.to_datetime(tmstmp)
data['mean'] = data[['Temp_in1', 'Temp_in2', 'Temp_in2']].astype(float).mean(1)

# Create dataset before the scaling
new_data=data.shift(periods=-1, fill_value=0)
y = new_data['mean'].iloc[:len(new_data.index)-1,].to_numpy(dtype=float)
X = data.iloc[:len(data.index)-1,].to_numpy(dtype=float)
y = y.reshape(len(y), 1)


# Define X, y
X, y = scaling(X, y)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state = 4)

model = Sequential()
model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(10, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
print(model.summary())


history = model.fit(x_train,y_train,epochs=60, batch_size=100, validation_data=(x_test,y_test), shuffle=False)

# Evaluate model
results = model.predict(x_train)

y_train, y_hat_train = descaling(x_train, y_train, results)

rmse = sqrt(mean_squared_error(y_train, y_hat_train))
print(f'Train: {rmse}')

results = model.predict(x_test)

y_test, y_hat_test = descaling(x_test, y_test, results)

rmse = sqrt(mean_squared_error(y_test, y_hat_test))
print(f'Test:  {rmse}')


#fig = plt.figure()
plt.xlabel('y')
plt.ylabel('y_hat')
plt.scatter(y_test,y_hat_test, s=1, label = 'y vs. y_hat')
plt.plot(y_test, y_test, 'r', label = 'y = y')
plt.legend()
plt.savefig('../fig/testregression.png')
plt.close()

#fig = plt.figure()
plt.xlabel('y')
plt.ylabel('y_hat')
plt.scatter(y_train,y_hat_train, s=1, label = 'y vs. y_hat')
plt.plot(y_train, y_train, 'r', label = 'y = y')
plt.legend()
plt.savefig('../fig/trainregression.png')
plt.close()

#fig = plt.figure()
plt.ylabel('MSE')
plt.xlabel("Epochs")
plt.plot(history.history['loss'], label = 'Test')
plt.plot(history.history['val_loss'], label = 'Train')
plt.legend()
plt.savefig('../fig/loss.png')
plt.close()

plot_model(model, to_file='../fig/model.png', show_shapes=True, show_layer_names=True)

"""Restart to estimate the whole data
"""

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
data['mean'] = data[['Temp_in1', 'Temp_in2', 'Temp_in2']].astype(float).mean(1)

# Create dataset before the scaling
new_data=data.shift(periods=-1, fill_value=0)
y = new_data['mean'].iloc[:len(new_data.index)-1,].to_numpy(dtype=float)
X = data.iloc[:len(data.index)-1,].to_numpy(dtype=float)
y = y.reshape(len(y), 1)

# Define X, y
X, y = scaling(X, y)
res = model.predict(X)
y, y_hat = descaling(X, y, res)

plt.plot(y, label='Observed Temperature')
plt.scatter(range(len(y_hat)), y_hat, color = 'r', s=4, label='Predictions')
plt.xlabel('Timesteps')
plt.ylabel('Internal Temperature [\u00B0C]')
plt.legend()
plt.savefig('../fig/predictions.png')
