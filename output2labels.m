function [labels] = output2labels(Y)
%returns the prediction as a one hotencode of the lable

[~, labels]	= max(Y);       % Get the index of the neuron with the maximal activity
labels      = labels' - 1;	% Correct for matlab indices (start with 1)

end

