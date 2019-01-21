clear all

filename = 'milk\monthly-milk-production-pounds-p.csv';
startRow = 2;

[inputMilk, outputMilk ]= load_Milk_DB(filename);
 inputMilk = posixtime(inputMilk);

%  network_milk;

filename = 'temperature\mean-daily-temperature-fisher-ri.csv';

[inputTemperature, outputTemperature ] = load_temperature_DB(filename,startRow);
 inputTemperature = posixtime(inputTemperature);
 
outputTemperature = strrep(outputTemperature, '?', '-');
outputTemperature = str2double(outputTemperature);


%  network_temperature;

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

network_dryer;

% % %Network DRYER
% % 
% % X = tonndata(input_dryer,false,false);
% % T = tonndata(output_dryer,false,false);
% % 
% % % Choose a Training Function
% % % For a list of all training functions type: help nntrain
% % % 'trainlm' is usually fastest.
% % % 'trainbr' takes longer but may be better for challenging problems.
% % % 'trainscg' uses less memory. Suitable in low memory situations.
% % trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% % 
% % % Create a Nonlinear Autoregressive Network with External Input
% % inputDelays = 1:2;
% % feedbackDelays = 1:2;
% % hiddenLayerSize = 10;
% % net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
% % 
% % % Prepare the Data for Training and Simulation
% % % The function PREPARETS prepares timeseries data for a particular network,
% % % shifting time by the minimum amount to fill input states and layer
% % % states. Using PREPARETS allows you to keep your original time series data
% % % unchanged, while easily customizing it for networks with differing
% % % numbers of delays, with open loop or closed loop feedback modes.
% % [x,xi,ai,t] = preparets(net,X,{},T);
% % 
% % % Setup Division of Data for Training, Validation, Testing
% % net.divideParam.trainRatio = 70/100;
% % net.divideParam.valRatio = 15/100;
% % net.divideParam.testRatio = 15/100;
% % net.divideFcn='divideblock';
% % 
% % net.trainParam.goal = 0;	    
% % net.trainParam.mu=1.0000e-003; 	
% % net.trainParam.mu_inc=10;		
% % net.trainParam.mu_dec=1; 		
% % net.trainParam.min_grad = 1.0000e-015 	
% % net.trainParam.epochs =5000;			
% % net.trainParam.max_fail=5000;
% % 
% % % Train the Network
% % [net,tr] = train(net,x,t,xi,ai);
% % 
% % % Test the Network
% % y = net(x,xi,ai);
% % e = gsubtract(t,y);
% % performance = perform(net,t,y)
% % 
% % % View the Network
% % view(net)
% % 
% % % Plots
% % % Uncomment these lines to enable various plots.
% % %figure, plotperform(tr)
% % %figure, plottrainstate(tr)
% % %figure, ploterrhist(e)
% % %figure, plotregression(t,y)
% % %figure, plotresponse(t,y)
% % %figure, ploterrcorr(e)
% % %figure, plotinerrcorr(x,e)
% % 
% % % Closed Loop Network
% % % Use this network to do multi-step prediction.
% % % The function CLOSELOOP replaces the feedback input with a direct
% % % connection from the outout layer.
% % netc = closeloop(net);
% % netc.name = [net.name ' - Closed Loop'];
% % view(netc)
% % [xc,xic,aic,tc] = preparets(netc,X,{},T);
% % yc = netc(xc,xic,aic);
% % closedLoopPerformance = perform(net,tc,yc)
% % 
% % % Step-Ahead Prediction Network
% % % For some applications it helps to get the prediction a timestep early.
% % % The original network returns predicted y(t+1) at the same time it is
% % % given y(t+1). For some applications such as decision making, it would
% % % help to have predicted y(t+1) once y(t) is available, but before the
% % % actual y(t+1) occurs. The network can be made to return its output a
% % % timestep early by removing one delay so that its minimal tap delay is now
% % % 0 instead of 1. The new network returns the same outputs as the original
% % % network, but outputs are shifted left one timestep.
% % nets = removedelay(net);
% % nets.name = [net.name ' - Predict One Step Ahead'];
% % view(nets)
% % [xs,xis,ais,ts] = preparets(nets,X,{},T);
% % ys = nets(xs,xis,ais);
% % stepAheadPerformance = perform(nets,ts,ys)
