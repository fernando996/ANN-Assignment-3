

x = input';
t = output';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = ["trainlm", "trainbr", "trainscg"];

% Create a Fitting Network
neurons = [4, 7, 10, 12, 15,  20];


% Create a Nonlinear Autoregressive Network with External Input
inputDelays = 1:2;
feedbackDelays = 1:2;
aux=1;

for i=1:1:size(neurons,2)
    
    for j=1:1:10
        for w=1:1:length(trainFcn)
            
            net = fitnet(neurons(i),trainFcn{w});

            net = setwb(net, -2.4 + (2.4+2.4)*rand(10,1));

            data_save.IW_initial{aux} = net.IW{1,1};
            data_save.b_initial{aux} = net.b{1};

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
            [net,tr] = train(net,x,t);
            
            data_save.net{aux} = net;
            data_save.tr{aux} = tr;
            data_save.x{aux}  = x ;
            data_save.t{aux}  = t ;

            % Test the Network
            y = net(x);
            data_save.y{aux} = y;
            data_save.e{aux} = gsubtract(t,y);
            
            data_save.performance{aux} = perform(net,t,y)

            
             A= [y' t'];
             data_save.R{aux} = corrcoef(A);
            % View the Network
%             view(net)
            clear net, tr, x,  t, y, A;
            aux = aux + 1;

        end
     

    end

end

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

