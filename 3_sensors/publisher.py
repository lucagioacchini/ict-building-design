import paho.mqtt.client as PahoMQTT
import time
import pandas as pd
import json
from datetime import datetime
import os

class MyPublisher:
	def __init__(self, clientID):
		self.clientID = clientID
		# create an instance of paho.mqtt.client
		self._paho_mqtt = PahoMQTT.Client(self.clientID, False) 
		# register the callback
		self._paho_mqtt.on_connect = self.myOnConnect
		self.messageBroker = 'localhost'

	def start (self):
		#manage connection to broker
		self._paho_mqtt.connect(self.messageBroker, 1883)
		self._paho_mqtt.loop_start()

	def stop (self):
		self._paho_mqtt.loop_stop()
		self._paho_mqtt.disconnect()

	def myPublish(self, topic, message):
		# publish a message with a certain topic
		self._paho_mqtt.publish(topic, message, 2)

	def myOnConnect (self, paho_mqtt, userdata, flags, rc):
		print ("Connected to %s with result code: %d" % (self.messageBroker, rc))


meas = {
    'BLOCK1:ZONE3:Zone Operative Temperature [C](TimeStep)':'Zone3:Temp',
    'BLOCK1:ZONE2:Zone Operative Temperature [C](TimeStep)':'Zone2:Temp',
    'BLOCK1:ZONE1:Zone Operative Temperature [C](TimeStep)':'Zone1:Temp',
    'DistrictCooling:Facility [J](TimeStep)':'Cool',
    'DistrictHeating:Facility [J](TimeStep)':'Heat',
    'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)':'Outdoor:Temp',
    'Environment:Site Outdoor Air Barometric Pressure [Pa](TimeStep)':'Outdoor:Pressure',
    'Environment:Site Wind Speed [m/s](TimeStep)':'Wind',
    'BLOCK1:ZONE1:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)':'Zone1:SolarRad',
    'BLOCK1:ZONE2:Zone Windows Total Transmitted Solar Radiation Rate [W](TimeStep)':'Zone2:SolarRad',
    'BLOCK1:ZONE3:Zone Air Relative Humidity [%](TimeStep)':'Zone3:Humidity',
    'BLOCK1:ZONE2:Zone Air Relative Humidity [%](TimeStep)':'Zone2:Humidity',
    'BLOCK1:ZONE1:Zone Air Relative Humidity [%](TimeStep)':'Zone1:Humidity',
    'Electricity:Facility [J](TimeStep)':'Power'
}

pub = MyPublisher("MyPublisher")
pub.start()

df=pd.read_csv('../data/ensig/epraw.csv',sep=',',decimal=',',index_col=0)

GATEWAY_NAME="VirtualBuilding"
for i in range(len(df.index)):
    tt = df.index[i]
    days, hours = tt.split('  ')
    tt = f'{days}/2017{hours}'
    tt = tt.replace(' ', '')
    if '201724:' in tt:
        tt=tt.replace('24:', '00:')
        timestamp = time.mktime(datetime.strptime(tt, "%m/%d/%Y%H:%M:%S").timetuple())
        timestamp += 86400
    else:
        timestamp = time.mktime(datetime.strptime(tt, "%m/%d/%Y%H:%M:%S").timetuple())

    for key, val in meas.items():
        if 'District' in key or 'Electricity' in key:
            new_val = float(df[key][i])/3.6e6
        elif 'Pressure' in key:
            new_val = float(df[key][i])/1000
        else:
            new_val = float(df[key][i])
        payload={
            "location":str(GATEWAY_NAME),
            "measurement":val,
            "time_stamp":int(timestamp),
            "value":new_val
        }
        print(payload)
        pub.myPublish ('ict4bd_virtualB', json.dumps(payload)) 	
        time.sleep(.1)

pub.stop()
