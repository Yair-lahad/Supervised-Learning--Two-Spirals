function [err, acc] = evaluate_MLP(Y, Y0, labels,loss)
%EVALUATE_MLP Evaluate a MLP
%   Get the error and accuracy of a MLP on a given dataset according to
% the loss function given. 

% Get statistics
err         = loss(Y,Y0);      % squared error
Y_labels	= output2labels(Y);
acc         = mean(double(Y_labels' == labels));	% accuracy

end

