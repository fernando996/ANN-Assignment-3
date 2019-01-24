clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Milk DB                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Milk Load
filename = 'milk\monthly-milk-production-pounds-p.csv';
startRow = 2;

[inputMilk, outputMilk ]= load_Milk_DB(filename);

inputMilk = posixtime(inputMilk);
input = inputMilk;
ouput = outputMilk;

%Milk Train
network_narx;
%Milk Data Save
data_save_Milk= data_save;

%  network_milk;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              Temperature DB                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filename = 'temperature\mean-daily-temperature-fisher-ri.csv';

[inputTemperature, outputTemperature ] = load_temperature_DB(filename,startRow);
 inputTemperature = posixtime(inputTemperature);
 
outputTemperature = strrep(outputTemperature, '?', '-');
outputTemperature = str2double(outputTemperature);

input = inputTemperature;
ouput = outputTemperature;

clear data_save;
network_narx;

data_save_temperature= data_save;
%  network_temperature;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              Temperature DB                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Sampling:
% 	0.1 min
% Number:
% 	7500
% Inputs:
% 	q: Coolant Flow l/min
% Outputs:
% 	Ca: Concentration mol/l
% 	T: Temperature Kelvin degrees

filename = 'reactor\cstr.dat';

reactor_DB = load_reactor(filename);

input_reactor =  cell2mat(reactor_DB(:,[2:3]));
output_reactor = cell2mat(reactor_DB(:,4));
input = input_reactor;
output = output_reactor ;

clear data_save;
network_fitnet;

data_save_reactor= data_save;

% network_reactor;

% Sampling:
% Number:
% 	1000
% Inputs:
% 	u: voltage of the heating device 
% Outputs:

filename = 'dryer\dryer.dat';

dryer_DB = load_dryer_DB(filename);

input_dryer =  cell2mat(dryer_DB(:,1));
output_dryer = cell2mat(dryer_DB(:,2));

input = input_dryer;
output = output_dryer ;

clear data_save;
network_fitnet;

data_save_dryer= data_save;


%network_dryer;
