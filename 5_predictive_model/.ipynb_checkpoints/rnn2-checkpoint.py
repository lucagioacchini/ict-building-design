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

LAGS = 3


def scaleData(X_toScale, *y):
    split = .7

    scalers = {
        'x':[]
    }

    if len(y)>0:
        y = y[0]
        x_test = X_toScale[int(X_toScale.shape[0]*split):,:,:]
        x_train = X_toScale[:int(X_toScale.shape[0]*split),:,:]
        y_test = y[int(y.shape[0]*split):,:]
        y_train = y[:int(y.shape[0]*split),:]
        
        
        x_train_s = np.zeros((x_train.shape[0],x_train.shape[1],x_train.shape[2]))
        x_test_s = np.zeros((x_test.shape[0],x_test.shape[1],x_test.shape[2]))
        for i in range(X_toScale.shape[1]):
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))

            scaler_x.fit(x_train[:,i,:])

            x_train_s[:,i,:] = scaler_x.transform(x_train[:,i,:])
            x_test_s[:,i,:] = scaler_x.transform(x_test[:,i,:])

            with open(f'scaler{i}x.pkl', 'wb') as file:
                pickle.dump(scaler_x, file)
                print('Scaler saved')

        scaler_y.fit(y_train)
        y_train_s = np.zeros((y_train.shape))
        y_test_s = np.zeros((y_test.shape))
        
        y_train_s = scaler_y.transform(y_train)
        y_test_s = scaler_y.transform(y_test)

        y_test_s = y_test_s.reshape(y_test_s.shape[0],y_test_s.shape[1],1)
        y_train_s = y_train_s.reshape(y_train_s.shape[0],y_train_s.shape[1],1)
        
        with open('scalery.pkl', 'wb') as file:
            pickle.dump(scaler_y, file)
        print('Scaler saved')
        
        
        return x_train_s, y_train_s, x_test_s, y_test_s

    else:
        x_scaled = np.zeros((X_toScale.shape[0],X_toScale.shape[1],X_toScale.shape[2]))
        for i in range(X_toScale.shape[1]):
            with open(f'scaler{i}x.pkl', 'rb') as file:
                scaler_x = pickle.load(file)
                print('Scaler Loaded')
                x_scaled[:,i,:] = scaler_x.transform(X_toScale[:,i,:])
            
        return x_scaled


def descaleData(y_hat):
    with open('scalery.pkl', 'rb') as file:
        scaler_y = pickle.load(file)

    y_hat_r = y_hat.reshape(y_hat.shape[0],y_hat.shape[1])
    y_hat_d = scaler_y.inverse_transform(y_hat_r)

    return y_hat_d


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


def rollByLags(X_temp):
    lag = -LAGS
    y = np.roll(X_temp, lag, axis=0)[:,:,-1]
    X, y = X_temp[:lag,:,:], y[:lag,:]
    print('Rolled:')
    print(f'\tX:{X.shape}')
    print(f'\ty:{y.shape}')

    return X, y


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
data['mean'] = data[['Temp_in1', 'Temp_in2', 'Temp_in2']].astype(float).mean(1)

# +
temp = prepareData(data)
X, y = rollByLags(temp)

x_train, y_train, x_test, y_test = scaleData(X, y)

# +
model = Sequential()
model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True, unroll=True, stateful = False))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(80, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True, unroll=True, stateful = False)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', metrics=['mae', 'mape'], optimizer='adam')
print(model.summary())

history = model.fit(x_train,y_train, epochs=100, batch_size=50, validation_data=(x_test,y_test), shuffle=False)
# -





# +
plt.rcParams["figure.figsize"] = (10,7)
START=1000
END=2000

temp_df = prepareData(data[START:END])
val_y = temp_df[LAGS:,0,-1]
val_x = scaleData(temp_df)

res = model.predict(val_x)
val_y_hat = descaleData(res)

plt.rcParams["figure.figsize"] = (10,7)
plt.plot(val_y, label='Observations', color = 'orangered', linewidth=2)
plt.plot(val_y_hat[:,2], label='Predictions', color = 'steelblue', linewidth=2)
plt.xlabel('Timesteps')
plt.ylabel('Internal Average Temperature [\u00B0C]')
plt.legend()
plt.xlim(0,END-START-1)
plt.grid(linestyle='--')
plt.savefig('../fig/predictionsTemperature.png')
plt.close()

# +
START=0
END=data.shape[0]-LAGS

temp_df = prepareData(data[START:END])
val_y = temp_df[LAGS:,0,-1]
val_x = scaleData(temp_df)

res = model.predict(val_x)
val_y_hat = descaleData(res)

# -

plt.rcParams["figure.figsize"] = (10,7)
plt.plot(val_y, label='Observations', color = 'orangered', linewidth=2, alpha=.7)
plt.plot(val_y_hat[:,2], label='Predictions', color = 'steelblue', linewidth=2, alpha=.6)
plt.xlabel('Timesteps')
plt.ylabel('Internal Average Temperature [\u00B0C]')
plt.legend()
plt.xlim(0,END-START-1)
plt.grid(linestyle='--')
plt.savefig('../fig/fullpredictionsTemperature.png')
plt.close()

plt.rcParams["figure.figsize"] = (10,7)
plt.scatter(val_y, val_y_hat[:-LAGS,2], label='Predictions on Observations', color = 'steelblue', s=.5)
plt.plot(val_y,val_y, label='Regression Line', color = 'r')
plt.xlabel('Observations')
plt.ylabel('Predictions')
lgnd = plt.legend()
lgnd.legendHandles[1]._sizes = [10]
plt.savefig('../fig/regressionTemperature.png')
plt.close()

plt.rcParams["figure.figsize"] = (10,7)
plt.ylabel('MSE')
plt.xlabel("Epochs")
plt.plot(history.history['loss'], label = 'Train', color='steelblue', linewidth=2)
plt.plot(history.history['val_loss'], label = 'Test', color='orangered', linewidth=2)
plt.legend()
plt.grid(linestyle='--')
plt.xlim(-2,100)
plt.ylim(0)
plt.savefig('../fig/lossTemperature.png')
plt.close()
