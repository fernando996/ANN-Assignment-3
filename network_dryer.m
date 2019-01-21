
%Network DRYER

X = tonndata(input_dryer,false,false);
T = tonndata(output_dryer,false,false);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
%trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

neurons = [4, 7, 10, 12, 15,  20];
trainFcn = ["trainlm", "trainbr", "trainscg"];

% Create a Nonlinear Autoregressive Network with External Input
inputDelays = 1:2;
feedbackDelays = 1:2;
aux=1;

for i=1:1:size(neurons,2)
    
    for j=1:1:10
        for w=1:1:length(trainFcn)

            net = narxnet(inputDelays,feedbackDelays,neurons(i),'open',trainFcn{w});
            
            [x,xi,ai,t] = preparets(net,X,{},T);

            % Setup Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = 70/100;
            net.divideParam.valRatio = 15/100;
            net.divideParam.testRatio = 15/100;
            net.divideFcn='divideblock';

            net.trainParam.goal = 0;	    
            net.trainParam.mu=1.0000e-003; 	
            net.trainParam.mu_inc=10;		
            net.trainParam.mu_dec=1; 		
            net.trainParam.min_grad = 1.0000e-015 	
            net.trainParam.epochs =5000;			
            net.trainParam.max_fail=5000;

            % Train the Network
            [net,tr] = train(net,x,t,xi,ai);

            dryer_net{aux} = net;
            dryer_tr{aux} = tr;
            
            dryer_x{aux}  = x ;
            dryer_xi{aux}  = xi;
            dryer_ai{aux}  = ai;
            dryer_t{aux}  = t ;
            
            
            % Test the Network
            y = net(x,xi,ai);
            dryer_y{aux} = y;
            dryer_e{aux} = gsubtract(t,y);
            dryer_performance{aux} = perform(net,t,y)
            
            % Closed Loop Network
            % Use this network to do multi-step prediction.
            % The function CLOSELOOP replaces the feedback input with a direct
            % connection from the outout layer.
            netc= closeloop(net);
            dryer_netc{aux} = netc;
            dryer_netc{aux}.name = [dryer_net{aux}.name ' - Closed Loop'];

            [dryer_xc{aux},dryer_xic{aux},dryer_aic{aux},dryer_tc{aux}] = preparets( dryer_netc{aux},X,{},T);
            dryer_yc{aux} = netc(dryer_xc{aux},dryer_xic{aux},dryer_aic{aux});
            dryer_closedLoopPerformance{aux} = perform(dryer_net{aux},dryer_tc{aux},dryer_yc{aux})

            clear net, tr, x, xi, ai, t, y;
            aux = aux+1;
        end
     

    end

end









% % Step-Ahead Prediction Network
% % For some applications it helps to get the prediction a timestep early.
% % The original network returns predicted y(t+1) at the same time it is
% % given y(t+1). For some applications such as decision making, it would
% % help to have predicted y(t+1) once y(t) is available, but before the
% % actual y(t+1) occurs. The network can be made to return its output a
% % timestep early by removing one delay so that its minimal tap delay is now
% % 0 instead of 1. The new network returns the same outputs as the original
% % network, but outputs are shifted left one timestep.
% nets = removedelay(net);
% nets.name = [net.name ' - Predict One Step Ahead'];
% view(nets)
% [xs,xis,ais,ts] = preparets(nets,X,{},T);
% ys = nets(xs,xis,ais);
% stepAheadPerformance = perform(nets,ts,ys)
