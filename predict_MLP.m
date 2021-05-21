function [Y] = predict_MLP(Net, X)
%PREDICT_MLP Get the output of a MLP for a given dataset

% Get the MLP's output for the given set
s = X;                         	% set the input
for l = 1:length(Net)        	% forward pass
    
    s = Net(l).g(Net(l).W * [s;ones(1,size(s,2))]);	% get next layer's activities
end
Y = softmax(s);                         	% get the output

end

