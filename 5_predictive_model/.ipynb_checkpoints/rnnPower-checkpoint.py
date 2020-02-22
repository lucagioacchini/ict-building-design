# # LSTM model for HVAC Power Consumption Forecasting

import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, TimeDistributed, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import timedelta  
from keras.utils import plot_model
import pickle

df=pd.read_csv('../data/ensig/epraw.csv',sep=',',decimal=',',index_col=0, low_memory=False)

cols = [
    'BLOCK1:ZONE3:Zone Operative Temperature [C](TimeStep)',
    'BLOCK1:ZONE2:Zone Operative Temperature [C](TimeStep)',
    'BLOCK1:ZONE1:Zone Operative Temperature [C](TimeStep)',
    'DistrictCooling:Facility [J](TimeStep)',
    'DistrictHeating:Facility [J](TimeStep)',
]

meas = {
    'BLOCK1:ZONE3:Zone Operative Temperature [C](TimeStep)':'Temp_in3',
    'BLOCK1:ZONE2:Zone Operative Temperature [C](TimeStep)':'Temp_in2',
    'BLOCK1:ZONE1:Zone Operative Temperature [C](TimeStep)':'Temp_in1',
    'DistrictCooling:Facility [J](TimeStep)':'Cool',
    'DistrictHeating:Facility [J](TimeStep)':'Heat',
}

LAGS = 3


# Split the entire dataset into the training and test one. Do the same with the target array

def splitData(X_toScale, y):
    split = .6

    x_test = X_toScale[int(X_toScale.shape[0]*split):,:,:]
    x_train = X_toScale[:int(X_toScale.shape[0]*split),:,:]
    y_test = y[int(y.shape[0]*split):,:]
    y_train = y[:int(y.shape[0]*split),:]

    y_test = y_test.reshape(y_test.shape[0],y_test.shape[1],1)
    y_train = y_train.reshape(y_train.shape[0],y_train.shape[1],1)

    return x_train, y_train, x_test, y_test


# Reshape the dataset from NxF to NxTxF where N is the number of samples, F is the number of features and T is the number of timesteps

def prepareData(data):
    X = []
    mat_list = []
    cnt = 0
    print('Preparing data...')
    for index in data.index:
        # Create a new matrix representing 24h
        new_sample = np.zeros(
            (LAGS, data.shape[1]),
            dtype=float
        )

        to_insert = data.loc[index]
        mat_list.append({
            'matrix':new_sample,
            'filled_idx':-1
        })
        # Append the data referred to the current hour to all the matrix
        # by shifting its position
        for item in mat_list:
            item['filled_idx']+=1
            item['matrix'][item['filled_idx']] = to_insert

        # If a matrix is full move it to the X array
        for item in mat_list:
            if item['filled_idx'] == LAGS-1:
                X.append(mat_list.pop(0)['matrix'])  

        if cnt == LAGS-1:
            cnt = 0
        else:
            cnt+=1
    X = np.asarray(X)
    print('Data Prepared')

    return np.asarray(X)


# Create the target array by rolling back T times the samples. T is the number of timesteps.

def rollByLags(X_temp):
    lag = -LAGS
    y = np.roll(X_temp, lag, axis=0)[:,:,-1]
    X, y = X_temp[:lag,:,:], y[:lag,:]
    print('Rolled:')
    print(f'\tX:{X.shape}')
    print(f'\ty:{y.shape}')

    return X, y


# Create the binary arrays of the HVAC schedule.

def manageSchedule(data):
    date_stuff = {
        'Cool_stat':[],
        'Heat_stat':[]
    }
    prev_day = 0
    dow = 6
    for item in data.index:
        if item.day != prev_day:
            if dow == 7:
                dow = 0
            dow+=1
            prev_day = item.day
        if dow!=7:
            if item.hour >= 8 and item.hour <= 18:
                if item.month >= 5 and item.month <9 and dow!=6:
                    date_stuff['Cool_stat'].append(1)
                    date_stuff['Heat_stat'].append(0)
                elif item.month < 5 or item.month >=10:
                    date_stuff['Heat_stat'].append(1)
                    date_stuff['Cool_stat'].append(0)
                else:
                    date_stuff['Cool_stat'].append(0)
                    date_stuff['Heat_stat'].append(0)
            else:
                date_stuff['Cool_stat'].append(0)
                date_stuff['Heat_stat'].append(0)
        else:
            date_stuff['Cool_stat'].append(0)
            date_stuff['Heat_stat'].append(0)

    return date_stuff


# +
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
        if timestamp.year == 2018:
            timestamp -= timedelta(days=365)
            
    else:
        timestamp = datetime.strptime(tt, "%m/%d/%Y%H:%M:%S")
    tmstmp.append(timestamp)
data.index = pd.to_datetime(tmstmp)
data = data.sort_index()

# Retrieve the HVAC schedule
schedules = manageSchedule(data)
data['Cool_stat'] = schedules['Cool_stat']
data['Heat_stat'] = schedules['Heat_stat']
data['Stat'] = data['Cool_stat']+data['Heat_stat']
data = data.drop(columns=['Cool_stat', 'Heat_stat'])

# Manage the target variables
data['Cool'] = data['Cool']/3.6e6
data['Heat'] = data['Heat']/3.6e6
data['pwr'] = (data['Cool']+data['Heat'])
# -

# Resize the dataset, roll it and split into train and test ones

# +
temp = prepareData(data)
X, y = rollByLags(temp)

x_train, y_train, x_test, y_test = splitData(X, y)
# -

# Define the model

# +
model = Sequential()
model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True, unroll=True, stateful = False))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(80, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True, unroll=True, stateful = False)))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', metrics=['mae', 'mape'], optimizer='adam')
print(model.summary())

# Train the model
history = model.fit(x_train,y_train, epochs=100, batch_size=50, validation_data=(x_test,y_test), shuffle=False)
# -
# Plot the overestimations of the model

# +
plt.rcParams["figure.figsize"] = (10,7)
START=35000
END=40000

temp_df = prepareData(data[START:END])
val_y = temp_df[LAGS:,0,-1]
val_x = temp_df

val_y_hat = model.predict(val_x)

plt.rcParams["figure.figsize"] = (10,7)
plt.plot(val_y, label='Observations', color = 'orangered', linewidth=1.5)
plt.plot(val_y_hat[:,2], label='Predictions', color = 'steelblue',linestyle='-.', linewidth=1.5)
plt.xlabel('Timesteps')
plt.ylabel('HVAC Consumption [kWh]')
plt.legend()
plt.xlim(0,END-START-1)
plt.grid(linestyle='--')
plt.savefig('../fig/overPower.png', transparent=True)
plt.close()
# -

# Plot the underestimations of the model

# +
plt.rcParams["figure.figsize"] = (10,7)
START=250
END=2000

temp_df = prepareData(data[START:END])
val_y = temp_df[LAGS:,0,-1]
val_x = temp_df

val_y_hat = model.predict(val_x)

plt.rcParams["figure.figsize"] = (10,7)
plt.plot(val_y, label='Observations', color = 'orangered', linewidth=1.5)
plt.plot(val_y_hat[:,2], label='Predictions', color = 'steelblue',linestyle='-.', linewidth=1.5)
plt.xlabel('Timesteps')
plt.ylabel('HVAC Consumption [kWh]')
plt.legend()
plt.xlim(0,END-START-1)
plt.grid(linestyle='--')
plt.savefig('../fig/underPower.png', transparent=True)
plt.close()
# -

# Perform the evaluation on the entire dataset

# +
START=0
END=data.shape[0]-LAGS

temp_df = prepareData(data[START:END])
val_y = temp_df[LAGS:,0,-1]
val_x = temp_df

val_y_hat = model.predict(val_x)

# -

# Plot the obtained performances

plt.rcParams["figure.figsize"] = (10,7)
plt.plot(val_y, label='Observations', color = 'orangered', linewidth=2, alpha=.7)
plt.plot(val_y_hat[:,2], label='Predictions', color = 'steelblue', linewidth=2, alpha=.6)
plt.xlabel('Timesteps')
plt.ylabel('HVAC Consumption [kWh]')
plt.legend()
plt.xlim(0,END-START-1)
plt.grid(linestyle='--')
#plt.savefig('../fig/fullpredictionsPower.png')
#plt.close()

# Plot y_hat vs. y

plt.rcParams["figure.figsize"] = (10,7)
plt.scatter(val_y, val_y_hat[:-LAGS,2], label='Predictions on Observations', color = 'steelblue', s=.5)
plt.plot(val_y,val_y, label='Regression Line', color = 'r')
plt.xlabel('Observations')
plt.ylabel('Predictions')
lgnd = plt.legend()
lgnd.legendHandles[1]._sizes = [10]
plt.savefig('../fig/regressionPower.png')
plt.close()

# Plot the loss function

plt.rcParams["figure.figsize"] = (10,7)
plt.ylabel('MSE')
plt.xlabel("Epochs")
plt.plot(history.history['loss'], label = 'Train', color='steelblue',linestyle='-.', linewidth=2)
plt.plot(history.history['val_loss'], label = 'Test', color='orangered', linewidth=2)
plt.legend()
plt.grid(linestyle='--')
plt.xlim(-2,100)
plt.ylim(0)
plt.savefig('../fig/lossPower.png', transparent=True)
plt.close()
