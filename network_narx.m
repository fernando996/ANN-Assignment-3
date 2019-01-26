
%Network DRYER

X = tonndata(input,false,false);
T = tonndata(ouput,false,false);

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
            
%             net = setwb(net, rand(5));
            net = setwb(net, -2.4 + (2.4+2.4)*rand(5));
            net.inputs{1}.processFcns={'mapstd'};
            net.outputs{2}.processFcns={'mapstd'};


             data_save.IW_initial{aux} = net.IW{1,1};
             data_save.b_initial{aux} = net.b{1};
            
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

              data_save.net{aux} = net;
              data_save.tr{aux} = tr;
            
              data_save.x{aux}  = x ;
              data_save.xi{aux}  = xi;
              data_save.ai{aux}  = ai;
              data_save.t{aux}  = t ;
            
            
            % Test the Network
            y = net(x,xi,ai);
              data_save.y{aux} = y;
              data_save.e{aux} = gsubtract(t,y);
              data_save.performance{aux} = perform(net,t,y); %%MSE
            
            % Closed Loop Network
            % Use this network to do multi-step prediction.
            % The function CLOSELOOP replaces the feedback input with a direct
            % connection from the outout layer.
            netc= closeloop(net);
              data_save.netc{aux} = netc;
              data_save.netc{aux}.name = [  data_save.net{aux}.name ' - Closed Loop'];

            [  data_save.xc{aux},  data_save.xic{aux},  data_save.aic{aux},  data_save.tc{aux}] = preparets(   data_save.netc{aux},X,{},T);
              data_save.yc{aux} = netc(  data_save.xc{aux},  data_save.xic{aux},  data_save.aic{aux});
              data_save.closedLoopPerformance{aux} = perform(  data_save.net{aux},  data_save.tc{aux},  data_save.yc{aux});

            % Step-Ahead Prediction Network
            % For some applications it helps to get the prediction a timestep early.
            % The original network returns predicted y(t+1) at the same time it is
            % given y(t+1). For some applications such as decision making, it would
            % help to have predicted y(t+1) once y(t) is available, but before the
            % actual y(t+1) occurs. The network can be made to return its output a
            % timestep early by removing one delay so that its minimal tap delay is now
            % 0 instead of 1. The new network returns the same outputs as the original
            % network, but outputs are shifted left one timestep.
            nets = removedelay(net);
              data_save.nets{aux} = nets;
              data_save.nets{aux}.name = [  data_save.nets{aux}.name ' - Predict One Step Ahead'];
%             view(nets)
            [  data_save.xs{aux},  data_save.xis{aux},  data_save.ais{aux},  data_save.ts{aux}] = preparets(nets,X,{},T);
              data_save.ys{aux} = nets(  data_save.xs{aux},  data_save.xis{aux},  data_save.ais{aux});
              data_save.stepAheadPerformance{aux} = perform(nets,  data_save.ts{aux},  data_save.ys{aux});
            
            data_save.IW_final{aux} = net.IW{1,1};
            data_save.b_final{aux} = net.b{1};

            A= [cell2mat(y') cell2mat(t')];
             data_save.R{aux} = corrcoef(A)
   
            clear net, tr, x, xi, ai, t, y, netc,X,T,nets, A;
            aux = aux+1;
        end
     

    end

end









