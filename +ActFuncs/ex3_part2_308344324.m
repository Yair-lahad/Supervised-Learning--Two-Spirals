clear; close all; clc;

%% Parameters setting
% Network sizes
input_sz = 1;
hidden_sz = 500;
output_sz = 100;

% Connection ratio
connection_ratio = 0.5;
assert(connection_ratio <= 1 & connection_ratio >0, 'Connection ratio must be between 0 to 1')

% Activation function
activation = @ActFuncs.Tanh;
% Want to check the analytic part?
% activation = @(x) ((exp(3*x) - exp(-3*x)) ./ ((exp(x) + exp(-x)) .^3));

% Leakage variable
alpha = 1;
assert(alpha >= 0, 'Alpha must be non-negative')

% Regularization variable
beta = 0.0000001;

% Constant for weights between input to resovoir
gamma = 0.01;

% Constant for weights normalization in the resovoir
delta = 0.99;
% Want to check the analytic part?
% delta = delta * (4/3);
%% Initialize
% Generate randomly connected weights
K = [ones(1, round(hidden_sz * connection_ratio)), zeros(1, round(hidden_sz * (1 - connection_ratio)))];
K = repmat(K, hidden_sz, 1);
rand_order = randperm(hidden_sz ^ 2);
K = reshape(K(rand_order), hidden_sz, hidden_sz);

% Choose weights values randomly
K = K .* normrnd(0, 1, hidden_sz, hidden_sz);

% Normalize K by the largest real part eigen value
lambdas = max(abs(real(eig(K))));
K = (K / lambdas) * delta;

% Choose random weights between input to the hidden layer.
J = normrnd(0, 1, input_sz, hidden_sz) * gamma;


% Crate 10,000 sampels of guassion white noise
sampels_N = 10000;
input = randn(1, sampels_N);

r = zeros(1, hidden_sz);

%% Allocations
Y0_mat = zeros(output_sz, sampels_N - output_sz);
r_mat = zeros(hidden_sz + 2, sampels_N - output_sz);
r2_train = zeros(1, output_sz);
r2_test = zeros(1, output_sz);

%% Feed Forward
i = 0; % Index count initialize
for time_i = 1 : sampels_N - 1
    % For the first 100 steps just feed forward input, untill we can
    % generate Y0 properly
    X = input(time_i);
    r = (1 - alpha) * r + alpha * activation(X * J + r * K);
    % Getting the 100 output neurons teacher.
    if time_i >= output_sz
        i = i + 1; % Count index
        % Get the corresponding Y0
        Y0 = input(time_i : -1 : time_i - output_sz +1);
        % Reverse the order
        Y0_mat(:, i) = Y0(end : -1 : 1); 
        % Save the timestep corresponding network state (with bias and
        % input connection)
        r_mat(:, i) = [1, X, r];
    end
end

%% Getting the optimized weights from the resovior to the output
C = (1 / (sampels_N - output_sz)) *  (r_mat) * (r_mat)';
u = (1 / (sampels_N - output_sz)) * r_mat * Y0_mat';
% Add regularization
C_reg = C + beta * eye(size(C, 1));
% Compute the optimized W connections
W = C_reg \ u;

%% Getting the outputs per time step
    Y_mat = (r_mat' * W)';

% Computing r2 score per output neuron
for neuron_i = 1 : output_sz
    r2_train(neuron_i) = corr(Y_mat(neuron_i, :)', Y0_mat(neuron_i, :)') .^ 2;
end

%% Making new data set (Test)
input = randn(1, sampels_N);

% Feed Forward The new data
i = 0; % Index count initialize
for time_i = 1 : sampels_N - 1
    % For the first 100 steps just feed forward input, untill we can
    % generate Y0 properly
    X = input(time_i);
    r = (1 - alpha) * r + alpha * activation(X * J + r * K);
    % Getting the 100 output neurons teacher.
    if time_i >= output_sz
        i = i + 1; % Count index
        % Get the corresponding Y0
        Y0 = input(time_i : -1 : time_i - output_sz +1);
        % Reverse the order
        Y0_mat(:, i) = Y0(end : -1 : 1); 
        % Save the timestep corresponding network state (with bias and
        % input connection)
        r_mat(:, i) = [1, X, r];
    end
end

% Computing the Test outputs
Y_mat = (r_mat' * W)';

% Computing r2 score per output neuron
for neuron_i = 1 : output_sz
    r2_test(neuron_i) = corr(Y_mat(neuron_i, :)', Y0_mat(neuron_i, :)') .^ 2;
end

%% Plot And Results
figure('units','normalized', 'position', [0.25 ,0.3, 0.6, 0.6]);
plot(1 : 100, r2_train(end : -1 : 1))
hold on
plot(1 : 100, r2_test(end : -1 : 1))
title('R2 Score Between Y0 And Net Output Per Neuron')
xlabel('Neuron #')
ylabel('R2')
legend({'Train', 'Test'})

% Print the results to the command line
disp(['For training set:' newline 'Total network memory capacity is : ' num2str(sum(r2_train))])
disp('')
disp(['For test set:' newline 'Total network memory capacity is : ' num2str(sum(r2_test))])

