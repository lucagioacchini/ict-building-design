DIR=idf/8.6/*

for FILE in $DIR
do
  cd /usr/local/EnergyPlus-9-2-0/PreProcess/IDFVersionUpdater/
  ./Transition-V8-6-0-to-V8-7-0 /home/luca/Codes/ICT4BD/model/$FILE
  ./Transition-V8-7-0-to-V8-8-0 /home/luca/Codes/ICT4BD/model/$FILE
  ./Transition-V8-8-0-to-V8-9-0 /home/luca/Codes/ICT4BD/model/$FILE
  ./Transition-V8-9-0-to-V9-0-0 /home/luca/Codes/ICT4BD/model/$FILE
done
	

#cd /usr/local/EnergyPlus-9-2-0/PreProcess/IDFVersionUpdater/
#./Transition-V8-6-0-to-V8-7-0 /home/luca/Codes/ICT4BD/model/idf/8.6/test2.idf
#./Transition-V8-7-0-to-V8-8-0 /home/luca/Codes/ICT4BD/model/idf/8.6/test2.idf
#./Transition-V8-8-0-to-V8-9-0 /home/luca/Codes/ICT4BD/model/idf/8.6/test2.idf
#./Transition-V8-9-0-to-V9-0-0 /home/luca/Codes/ICT4BD/model/idf/8.6/test2.idf

#./Transition-V8-6-0-to-V8-7-0 /home/luca/Codes/ICT4BD/model/idf/8.6/DblArgNatO2.idf
#./Transition-V8-7-0-to-V8-8-0 /home/luca/Codes/ICT4BD/model/idf/8.6/DblArgNatO2.idf
#./Transition-V8-8-0-to-V8-9-0 /home/luca/Codes/ICT4BD/model/idf/8.6/DblArgNatO2.idf
#./Transition-V8-9-0-to-V9-0-0 /home/luca/Codes/ICT4BD/model/idf/8.6/DblArgNatO2.idf

#./Transition-V8-6-0-to-V8-7-0 /home/luca/Codes/ICT4BD/model/idf/8.6/SglNatO2.idf
#./Transition-V8-7-0-to-V8-8-0 /home/luca/Codes/ICT4BD/model/idf/8.6/SglNatO2.idf
#./Transition-V8-8-0-to-V8-9-0 /home/luca/Codes/ICT4BD/model/idf/8.6/SglNatO2.idf
#./Transition-V8-9-0-to-V9-0-0 /home/luca/Codes/ICT4BD/model/idf/8.6/SglNatO2.idf

#./Transition-V8-6-0-to-V8-7-0 /home/luca/Codes/ICT4BD/model/idf/8.6/TrpArgNatO2.idf
#./Transition-V8-7-0-to-V8-8-0 /home/luca/Codes/ICT4BD/model/idf/8.6/TrpArgNatO2.idf
#./Transition-V8-8-0-to-V8-9-0 /home/luca/Codes/ICT4BD/model/idf/8.6/TrpArgNatO2.idf
#./Transition-V8-9-0-to-V9-0-0 /home/luca/Codes/ICT4BD/model/idf/8.6/TrpArgNatO2.idf
