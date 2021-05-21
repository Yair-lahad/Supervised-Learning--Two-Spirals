%% Exercise 2- Neural Networks "Two Spirals" Problem
% A MLP learning which classify points to the corret spiral it belongs 
% using Neural Networks and Back propogation technique 

 % yair lahad 205493018
 clear;
 clc;

 %% Preprocess the data
[X_train, Y_train,train_labels]=preprocess_data(load('./Data/DATA_TRAIN.csv'));
[X_valid, Y_valid,valid_labels] = preprocess_data(load('./Data/DATA_valid.csv'));
% load test sets
% enter test data, uncomment next line and the *entire* Result section

%[X_test, Y_test,test_labels] = preprocess_data(load('insert test data csv'));

%% Parameters
% Training parameteres
eta         = 1e-3;	% learning rate
n_epochs    = 100;	% number of training epochs
batchSize  = 50;	% mini-batch size
% Net patameters:
%cost function             
costFunc = @crossentropy;     
% activation functions:
% can manipulate layers Activation functions at +ActFuncs folder
g_funcs = {@ActFuncs.ActCos,@ActFuncs.ReLU,@ActFuncs.Linear};
InitFunc = @InitFuncs.HeInit;         %weights init technique
% InitFunc = @InitFuncs.XavierInit;
%moment_var
alpha = 0.2;    %alpha=0 cancels momentum effect, alpha=0.2 is faster
%regularization
lambda=0.5; 
Ltype=1;    % 0- dont use , 1 for L1 and 2 for L2

%% Build and initialize the MLP
% Network structure
%NOTE:  number of activation functions need to match the order and 
%       number of layers (hiden_size + 2)
hiddenSize = [30,22];           %number of neurons per layer
% tried [30,30],[18,18],[25,25] and more - less efficient
inputSize = size(X_train,1);
outputSize = 2;
N = [inputSize, hiddenSize, outputSize];	% number of neurons per layer
L = length(N) - 1;	% number of layers
Net = arrayfun(@(n, n1, g) struct('W', InitFunc(n)*randn(n1, n+1), ...	% weights
                                  'g', g), ...                  % activation function
               [N(1:L)], N(2:L + 1), g_funcs);
%% Train the MLP
% Training statsitics per epoch
history = cell(n_epochs + 1, 1);
% Get metrics for the training and validation sets
[t_err, t_acc] = predict_evaluate_MLP(Net, X_train, Y_train, train_labels, costFunc);
[v_err, v_acc] = predict_evaluate_MLP(Net, X_valid, Y_valid, valid_labels, costFunc);
% Save statistics history
history{1} = struct('train_err', t_err, ... % training error
                    'train_acc', t_acc, ... % training accuracy
                    'valid_err', v_err, ... % validation error
                    'valid_acc', v_acc);    % validation accuracy
% Command log
fprintf('Epoch %d/%d, error = %0.3g, accuracy = %2.1f%%. \n', ...
        0, n_epochs, v_err, 100*v_acc);
    
dW = cell(1,L);
[dW{:}] = deal(0);
% Loop over epochs
for epoch = 1:n_epochs
    % Get a random order of the samples
    perm = randperm(size(X_train, 2));
    % Loop over all training samples (in mini-batches)
    for batch_start = 1:batchSize:length(perm)
        % Get the samples' indices for the current batch
        batch_end = min(batch_start + batchSize - 1, length(perm));
        batch_ind = batch_start:batch_end;
        % Get the current batch data
        X   = X_train(:, perm(batch_ind));
        Y0  = Y_train(:, perm(batch_ind));
        % Temporary neurons activities per layer
        s = cell(L + 1, 1);
        % Forward pass
        % NOTE: The layers' activities and derivatives of the current 
        %       mini-batch are stored in matrices, such that each column
        %       represents a different sample. 
        s{1} = struct('act', X, ...                     % set the input
                      'der', zeros(size(X)));           % for completeness
        for l = 1:L
            s{l}.act(size(s{l}.act,1)+1,:)=ones(1,size(s{l}.act,2));
            [g, gp] = Net(l).g(Net(l).W * s{l}.act);	% get next layer's activities 
                                                        % (and derivatives)
            s{l+1}  = struct('act', g, ...
                             'der', gp);                % save results per layer
        end
        Y = s{L + 1}.act;                               % get the output
        % according to class when we use cross entropy its better to use
        % softmax activation function for the output 
        Y=softmax(Y);
        % Back propagation
        delta = (Y - Y0).*s{L + 1}.der;
        for l = L:-1:1                     
            dW{l} = -(eta).*(delta * s{l}.act')+(dW{l}*alpha);    % get the weights update+momentum                           
            delta = ((Net(l).W(:,(1:end-1)))'*delta).*s{l}.der;   % update delta
             %regularization
             regu=0;
            if Ltype==1
                regu=(-eta/batchSize).*(lambda*sign(Net(l).W));  % L1
            elseif Ltype==2
               regu=(-eta/batchSize).*(lambda*Net(l).W);       % L2
            end
            Net(l).W = Net(l).W + dW{l} + regu;	                  % update the weights          
        end      
    end % mini-batches loop
    
    % Get metrics for the training and validation sets
    [t_err, t_acc] = predict_evaluate_MLP(Net, ...
                                          X_train, Y_train, train_labels,costFunc);
    [v_err, v_acc] = predict_evaluate_MLP(Net, ...
                                          X_valid, Y_valid, valid_labels, costFunc);
    % Save statistics history
    history{epoch + 1} = struct('train_err', t_err, ... % training error
                                'train_acc', t_acc, ... % training accuracy
                                'valid_err', v_err, ... % validation error
                                'valid_acc', v_acc);    % validation accuracy
    % Command log
    fprintf('Epoch %d/%d, error = %0.3g, accuracy = %2.1f%%\n', ...
            epoch, n_epochs, v_err, 100*v_acc);
    
end % epochs loop

%% Results for Test set
% Y_out                   = predict_MLP(Net, X_test);	% MLP's output
% Y_labels                = output2labels(Y_out);     % MLP's labels
% [test_err, test_acc]    = evaluate_MLP(Y_out, Y_test, test_labels,costFunc);
% fprintf('Test: error = %0.3g, accuracy = %2.1f%%. \n', ...
%         test_err, 100*test_acc);
% 
