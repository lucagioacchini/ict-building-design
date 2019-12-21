import pandas as pd 
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
PATH = '../data/eval/'

for fname in os.listdir(PATH):
    title = fname.replace('.csv', '')

    data = pd.read_csv(PATH+'/'+fname)
    #============================================================
    # Divide dataframes wrt WWR and sort wrt Insulation Thickness
    #============================================================
    wwr1_0 = [data.loc[x,:] for x in range(len(data.index)) if data['Window to Wall Ratio'][x]==.15 and data['Glazing1'][x]==.007]
    wwr2_0 = [data.loc[x,:] for x in range(len(data.index)) if data['Window to Wall Ratio'][x]==.5 and data['Glazing1'][x]==.007]
    wwr3_0 = [data.loc[x,:] for x in range(len(data.index)) if data['Window to Wall Ratio'][x]==.9 and data['Glazing1'][x]==.007]

    wwr1_2 = [data.loc[x,:] for x in range(len(data.index)) if data['Window to Wall Ratio'][x]==.15 and data['Glazing1'][x]==0.0285]
    wwr2_2 = [data.loc[x,:] for x in range(len(data.index)) if data['Window to Wall Ratio'][x]==.5 and data['Glazing1'][x]==0.0285]
    wwr3_2 = [data.loc[x,:] for x in range(len(data.index)) if data['Window to Wall Ratio'][x]==.9 and data['Glazing1'][x]==0.0285]

    wwr1_5 = [data.loc[x,:] for x in range(len(data.index)) if data['Window to Wall Ratio'][x]==.15 and data['Glazing1'][x]==.05]
    wwr2_5 = [data.loc[x,:] for x in range(len(data.index)) if data['Window to Wall Ratio'][x]==.5 and data['Glazing1'][x]==.05]
    wwr3_5 = [data.loc[x,:] for x in range(len(data.index)) if data['Window to Wall Ratio'][x]==.9 and data['Glazing1'][x]==.05]

    wwr1_0 = pd.DataFrame(wwr1_0)
    wwr2_0 = pd.DataFrame(wwr2_0)
    wwr3_0 = pd.DataFrame(wwr3_0)

    wwr1_2 = pd.DataFrame(wwr1_2)
    wwr2_2 = pd.DataFrame(wwr2_2)
    wwr3_2 = pd.DataFrame(wwr3_2)
    
    wwr1_5 = pd.DataFrame(wwr1_5)
    wwr2_5 = pd.DataFrame(wwr2_5)
    wwr3_5 = pd.DataFrame(wwr3_5)

    #=================================================
    # Plot Energy Consumption wrt Insulation Thickness
    #=================================================
    try:
        fig=plt.figure(figsize=(8, 6))
        # Annual
        plt.plot(wwr1_0['Insulation Thickness'], wwr1_0['TotalConsumption'], label='WWR=15%,Glaz=0.7cm', linewidth = .6, color='r')
        plt.plot(wwr2_0['Insulation Thickness'], wwr2_0['TotalConsumption'], label='WWR=50%,Glaz=0.7cm', linewidth = .6, color='g')
        plt.plot(wwr3_0['Insulation Thickness'], wwr3_0['TotalConsumption'], label='WWR=90%,Glaz=0.7cm', linewidth = .6, color='b')
        
        plt.plot(wwr1_2['Insulation Thickness'], wwr1_2['TotalConsumption'], label='WWR=15%,Glaz=2.85cm', linewidth = .8, linestyle='--', color='r')
        plt.plot(wwr2_2['Insulation Thickness'], wwr2_2['TotalConsumption'], label='WWR=50%,Glaz=2.85cm', linewidth = .8, linestyle='--', color='g')
        plt.plot(wwr3_2['Insulation Thickness'], wwr3_2['TotalConsumption'], label='WWR=90%,Glaz=2.85cm', linewidth = .8, linestyle='--', color='b')
        
        plt.plot(wwr1_5['Insulation Thickness'], wwr1_5['TotalConsumption'], label='WWR=15%,Glaz=5cm', linewidth = 1, linestyle=':', color='r')
        plt.plot(wwr2_5['Insulation Thickness'], wwr2_5['TotalConsumption'], label='WWR=50%,Glaz=5cm', linewidth = 1, linestyle=':', color='g')
        plt.plot(wwr3_5['Insulation Thickness'], wwr3_5['TotalConsumption'], label='WWR=90%,Glaz=5cm', linewidth = 1, linestyle=':', color='b')

        plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.17))
        plt.xlabel('Insulation Thickness [cm]')
        plt.ylabel('Energy Consumption [kWh]')
        plt.grid(linestyle='--', alpha=.5)

        plt.savefig(f"../fig/Ann{title}.png")

        fig=plt.figure(figsize=(8, 6))
        
        # Summer
        plt.plot(wwr1_0['Insulation Thickness'], wwr1_0['DistrictCooling:Facility'], label='WWR=15%,Glaz=0.7cm', linewidth = .6, color='r')
        plt.plot(wwr2_0['Insulation Thickness'], wwr2_0['DistrictCooling:Facility'], label='WWR=50%,Glaz=0.7cm', linewidth = .6, color='g')
        plt.plot(wwr3_0['Insulation Thickness'], wwr3_0['DistrictCooling:Facility'], label='WWR=90%,Glaz=0.7cm', linewidth = .6, color='b')
        
        plt.plot(wwr1_2['Insulation Thickness'], wwr1_2['DistrictCooling:Facility'], label='WWR=15%,Glaz=2.85cm', linewidth = .8, linestyle='--', color='r')
        plt.plot(wwr2_2['Insulation Thickness'], wwr2_2['DistrictCooling:Facility'], label='WWR=50%,Glaz=2.85cm', linewidth = .8, linestyle='--', color='g')
        plt.plot(wwr3_2['Insulation Thickness'], wwr3_2['DistrictCooling:Facility'], label='WWR=90%,Glaz=2.85cm', linewidth = .8, linestyle='--', color='b')
        
        plt.plot(wwr1_5['Insulation Thickness'], wwr1_5['DistrictCooling:Facility'], label='WWR=15%,Glaz=5cm', linewidth = 1, linestyle=':', color='r')
        plt.plot(wwr2_5['Insulation Thickness'], wwr2_5['DistrictCooling:Facility'], label='WWR=50%,Glaz=5cm', linewidth = 1, linestyle=':', color='g')
        plt.plot(wwr3_5['Insulation Thickness'], wwr3_5['DistrictCooling:Facility'], label='WWR=90%,Glaz=5cm', linewidth = 1, linestyle=':', color='b')

        plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.17))
        plt.xlabel('Insulation Thickness [cm]')
        plt.ylabel('Energy Consumption [kWh]')
        plt.grid(linestyle='--', alpha=.5)

        plt.savefig(f"../fig/Sum{title}.png")

        fig=plt.figure(figsize=(8, 6))
        
        # Winter
        plt.plot(wwr1_0['Insulation Thickness'], wwr1_0['DistrictHeating:Facility'], label='WWR=15%,Glaz=0.7cm', linewidth = .6, color='r')
        plt.plot(wwr2_0['Insulation Thickness'], wwr2_0['DistrictHeating:Facility'], label='WWR=50%,Glaz=0.7cm', linewidth = .6, color='g')
        plt.plot(wwr3_0['Insulation Thickness'], wwr3_0['DistrictHeating:Facility'], label='WWR=90%,Glaz=0.7cm', linewidth = .6, color='b')
        
        plt.plot(wwr1_2['Insulation Thickness'], wwr1_2['DistrictHeating:Facility'], label='WWR=15%,Glaz=2.85cm', linewidth = .8, linestyle='--', color='r')
        plt.plot(wwr2_2['Insulation Thickness'], wwr2_2['DistrictHeating:Facility'], label='WWR=50%,Glaz=2.85cm', linewidth = .8, linestyle='--', color='g')
        plt.plot(wwr3_2['Insulation Thickness'], wwr3_2['DistrictHeating:Facility'], label='WWR=90%,Glaz=2.85cm', linewidth = .8, linestyle='--', color='b')
        
        plt.plot(wwr1_5['Insulation Thickness'], wwr1_5['DistrictHeating:Facility'], label='WWR=15%,Glaz=5cm', linewidth = 1, linestyle=':', color='r')
        plt.plot(wwr2_5['Insulation Thickness'], wwr2_5['DistrictHeating:Facility'], label='WWR=50%,Glaz=5cm', linewidth = 1, linestyle=':', color='g')
        plt.plot(wwr3_5['Insulation Thickness'], wwr3_5['DistrictHeating:Facility'], label='WWR=90%,Glaz=5cm', linewidth = 1, linestyle=':', color='b')

        plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.17))
        plt.xlabel('Insulation Thickness [cm]')
        plt.ylabel('Energy Consumption [kWh]')
        plt.grid(linestyle='--', alpha=.5)

        plt.savefig(f"../fig/Win{title}.png")
    
    except:
        pass