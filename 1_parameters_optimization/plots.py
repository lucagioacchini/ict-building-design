import pandas as pd 
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
PATH = '../data/eval/'


def uVal(thick):
    """Compute the wals U-value starting from the thickness of 
    the insulation layer
    
    Parameters
    ----------
    thick : float
        insulation thickness in m
    """
    R_th = .17
    layers = [
		{
			'Thickness':.025,
			'Conductivity':.33
		},{
			'Thickness':thick,
			'Conductivity':.047
		},{
			'Thickness':.2,
			'Conductivity':.51
		},{
			'Thickness':.013,
			'Conductivity':.4
		},
	]
	
    res = R_th
    for item in layers:
        res+=item['Thickness']/item['Conductivity']
	
    return 1/res
	
for fname in os.listdir(PATH):
    title = fname.replace('.csv', '')
    if 'V_' not in title:
        data = pd.read_csv(PATH+'/'+fname)
        
        # Divide the dataframes wrt Argon Filled area and WWR
        wwr_15 = data.where(data['Window to Wall Ratio']==.15).dropna()
        wwr_15_1 = wwr_15.where(wwr_15['Argon1']<0.004).dropna()
        wwr_15_2 = wwr_15.where(wwr_15['Argon1']>0.004).dropna()
        wwr_15_2 = wwr_15_2.where(wwr_15_2['Argon1']<0.006).dropna()
        wwr_15_3 = wwr_15.where(wwr_15['Argon1']>0.01).dropna()

        wwr_50 = data.where(data['Window to Wall Ratio']==.50).dropna()
        wwr_50_1 = wwr_50.where(wwr_50['Argon1']<0.004).dropna()
        wwr_50_2 = wwr_50.where(wwr_50['Argon1']>0.004).dropna()
        wwr_50_2 = wwr_50_2.where(wwr_50_2['Argon1']<0.006).dropna()
        wwr_50_3 = wwr_50.where(wwr_50['Argon1']>0.01).dropna()

        wwr_90 = data.where(data['Window to Wall Ratio']==.90).dropna()
        wwr_90_1 = wwr_90.where(wwr_90['Argon1']<0.004).dropna()
        wwr_90_2 = wwr_90.where(wwr_90['Argon1']>0.004).dropna()
        wwr_90_2 = wwr_90_2.where(wwr_90_2['Argon1']<0.006).dropna()
        wwr_90_3 = wwr_90.where(wwr_90['Argon1']>0.01).dropna()
		
        u_vals = []
        for item in [.05,.1,.15,.2,.25,.3,.35]:
            u_vals.append(round(uVal(item),2))
		
        #=================================================
        # Plot Energy Consumption wrt Insulation Thickness
        #=================================================
        fig=plt.figure(figsize=(8, 6))
        # Annual
        plt.plot(wwr_15_1['Insulation Thickness'], wwr_15_1['TotalConsumption']/1e3, label='WWR=15%,ArThick=0.3cm', color='r')
        plt.plot(wwr_50_1['Insulation Thickness'], wwr_50_1['TotalConsumption']/1e3, label='WWR=50%,ArThick=0.3cm', color='g')
        plt.plot(wwr_90_1['Insulation Thickness'], wwr_90_1['TotalConsumption']/1e3, label='WWR=90%,ArThick=0.3cm', color='b')
        
        plt.plot(wwr_15_2['Insulation Thickness'], wwr_15_2['TotalConsumption']/1e3, label='WWR=15%,ArThick=0.5cm', linestyle='--', color='r')
        plt.plot(wwr_50_2['Insulation Thickness'], wwr_50_2['TotalConsumption']/1e3, label='WWR=50%,ArThick=0.5cm', linestyle='--', color='g')
        plt.plot(wwr_90_2['Insulation Thickness'], wwr_90_2['TotalConsumption']/1e3, label='WWR=90%,ArThick=0.5cm', linestyle='--', color='b')
        
        plt.plot(wwr_15_3['Insulation Thickness'], wwr_15_3['TotalConsumption']/1e3, label='WWR=15%,ArThick=2.3cm', linestyle=':', color='r')
        plt.plot(wwr_50_3['Insulation Thickness'], wwr_50_3['TotalConsumption']/1e3, label='WWR=50%,ArThick=2.3cm', linestyle=':', color='g')
        plt.plot(wwr_90_3['Insulation Thickness'], wwr_90_3['TotalConsumption']/1e3, label='WWR=90%,ArThick=2.3cm', linestyle=':', color='b')

        plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.30))
        plt.xlim(0.04,.35)
        plt.xlabel('Insulation Thickness [m]')
        plt.ylabel('Energy Consumption [MWh]')
        plt.grid(linestyle='--', alpha=.5)
        plt.twiny()
        plt.xlim(0.04,.35)
        plt.xticks(wwr_15_3['Insulation Thickness'], u_vals)
        plt.xlabel('Wall U-Value')
        
        
        plt.savefig(f"../fig/Ann{title}.png")
        plt.close()

        fig=plt.figure(figsize=(8, 6))
        
        # Summer
        plt.plot(wwr_15_1['Insulation Thickness'], wwr_15_1['DistrictCooling:Facility']/1e3, label='WWR=15%,ArThick=0.3cm', color='r')
        plt.plot(wwr_50_1['Insulation Thickness'], wwr_50_1['DistrictCooling:Facility']/1e3, label='WWR=50%,ArThick=0.3cm', color='g')
        plt.plot(wwr_90_1['Insulation Thickness'], wwr_90_1['DistrictCooling:Facility']/1e3, label='WWR=90%,ArThick=0.3cm', color='b')
        
        plt.plot(wwr_15_2['Insulation Thickness'], wwr_15_2['DistrictCooling:Facility']/1e3, label='WWR=15%,ArThick=0.5cm', linestyle='--', color='r')
        plt.plot(wwr_50_2['Insulation Thickness'], wwr_50_2['DistrictCooling:Facility']/1e3, label='WWR=50%,ArThick=0.5cm', linestyle='--', color='g')
        plt.plot(wwr_90_2['Insulation Thickness'], wwr_90_2['DistrictCooling:Facility']/1e3, label='WWR=90%,ArThick=0.5cm', linestyle='--', color='b')
        
        plt.plot(wwr_15_3['Insulation Thickness'], wwr_15_3['DistrictCooling:Facility']/1e3, label='WWR=15%,ArThick=2.3cm', linestyle=':', color='r')
        plt.plot(wwr_50_3['Insulation Thickness'], wwr_50_3['DistrictCooling:Facility']/1e3, label='WWR=50%,ArThick=2.3cm', linestyle=':', color='g')
        plt.plot(wwr_90_3['Insulation Thickness'], wwr_90_3['DistrictCooling:Facility']/1e3, label='WWR=90%,ArThick=2.3cm', linestyle=':', color='b')

        plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.30))
        plt.xlim(0.04,.35)
        plt.xlabel('Insulation Thickness [m]')
        plt.ylabel('Energy Consumption [MWh]')
        plt.grid(linestyle='--', alpha=.5)
        plt.twiny()
        plt.xlim(0.04,.35)
        plt.xticks(wwr_15_3['Insulation Thickness'], u_vals)
        plt.xlabel('Wall U-Value')
        
        
        plt.savefig(f"../fig/Sum{title}.png", transparent=True)
        plt.close()

        fig=plt.figure(figsize=(8, 6))
        
        # Winter
        plt.plot(wwr_15_1['Insulation Thickness'], wwr_15_1['DistrictHeating:Facility']/1e3, label='WWR=15%,ArThick=0.3cm', color='r')
        plt.plot(wwr_50_1['Insulation Thickness'], wwr_50_1['DistrictHeating:Facility']/1e3, label='WWR=50%,ArThick=0.3cm', color='g')
        plt.plot(wwr_90_1['Insulation Thickness'], wwr_90_1['DistrictHeating:Facility']/1e3, label='WWR=90%,ArThick=0.3cm', color='b')
        
        plt.plot(wwr_15_2['Insulation Thickness'], wwr_15_2['DistrictHeating:Facility']/1e3, label='WWR=15%,ArThick=0.5cm', linestyle='--', color='r')
        plt.plot(wwr_50_2['Insulation Thickness'], wwr_50_2['DistrictHeating:Facility']/1e3, label='WWR=50%,ArThick=0.5cm', linestyle='--', color='g')
        plt.plot(wwr_90_2['Insulation Thickness'], wwr_90_2['DistrictHeating:Facility']/1e3, label='WWR=90%,ArThick=0.5cm', linestyle='--', color='b')
        
        plt.plot(wwr_15_3['Insulation Thickness'], wwr_15_3['DistrictHeating:Facility']/1e3, label='WWR=15%,ArThick=2.3cm', linestyle=':', color='r')
        plt.plot(wwr_50_3['Insulation Thickness'], wwr_50_3['DistrictHeating:Facility']/1e3, label='WWR=50%,ArThick=2.3cm', linestyle=':', color='g')
        plt.plot(wwr_90_3['Insulation Thickness'], wwr_90_3['DistrictHeating:Facility']/1e3, label='WWR=90%,ArThick=2.3cm', linestyle=':', color='b')

        plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.30))
        plt.xlim(0.04,.35)
        plt.xlabel('Insulation Thickness [m]')
        plt.ylabel('Energy Consumption [MWh]')
        plt.grid(linestyle='--', alpha=.5)
        plt.twiny()
        plt.xlim(0.04,.35)
        plt.xticks(wwr_15_3['Insulation Thickness'], u_vals)
        plt.xlabel('Wall U-Value')
        
        
        plt.savefig(f"../fig/Win{title}.png", transparent=True)
        plt.close()

#=================================================
# Plot Lighting Consumption wrt Solar Visibility
#=================================================
plt.figure()  
for fname in os.listdir(PATH):
    title = fname.replace('.csv', '')
    if 'modelE' in title and 'V_' in title:
        data = pd.read_csv(PATH+'/'+fname)
        #============================================================
        # Divide dataframes wrt WWR and sort wrt Solar Visibility
        #============================================================
        wwr_15 = data.where(data['Window to Wall Ratio']==.15).dropna()
        wwr_50 = data.where(data['Window to Wall Ratio']==.50).dropna()
        wwr_90 = data.where(data['Window to Wall Ratio']==.90).dropna()

        plt.plot(wwr_15.Visibility1, wwr_15['InteriorLights:Electricity']/1e3, label='East, WWR=15%', color='r')
        plt.plot(wwr_15.Visibility1, wwr_50['InteriorLights:Electricity']/1e3, label='East, WWR=50%', color='g')
        plt.plot(wwr_15.Visibility1, wwr_90['InteriorLights:Electricity']/1e3, label='East, WWR=90%', color='b')
    elif 'modelN' in title and 'V_' in title:
        data = pd.read_csv(PATH+'/'+fname)
        #============================================================
        # Divide dataframes wrt WWR and sort wrt Solar Visibility
        #============================================================
        wwr_15 = data.where(data['Window to Wall Ratio']==.15).dropna()
        wwr_50 = data.where(data['Window to Wall Ratio']==.50).dropna()
        wwr_90 = data.where(data['Window to Wall Ratio']==.90).dropna()

        plt.plot(wwr_15.Visibility1, wwr_15['InteriorLights:Electricity']/1e3, label='North, WWR=15%', linestyle='--', color='r')
        plt.plot(wwr_15.Visibility1, wwr_50['InteriorLights:Electricity']/1e3, label='North, WWR=50%', linestyle='--', color='g')
        plt.plot(wwr_15.Visibility1, wwr_90['InteriorLights:Electricity']/1e3, label='North, WWR=90%', linestyle='--', color='b')
plt.legend()
plt.xlabel('Solar Visibility')
plt.ylabel('Lighting Consumption [MWh]')
plt.grid(linestyle='--', alpha=.5)  

plt.savefig(f"../fig/Visibility.png", transparent=True)
plt.close()
