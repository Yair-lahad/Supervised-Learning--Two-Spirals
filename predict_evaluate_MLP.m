function [err, acc] = predict_evaluate_MLP(Net, X, Y, labels, loss)
%PREDICT_EVALUATE_MLP Predicts then evaluates a MLP

% Get metrics for the given dataset
Y_out       = predict_MLP(Net, X);
[err, acc]  = evaluate_MLP(Y_out, Y, labels, loss);

end

