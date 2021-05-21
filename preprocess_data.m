function [X, Y,labels] = preprocess_data(data)
% preprocess the data

X = data(:,1:2)';
labels = data(:,3)';
Y = onehotencode(categorical(labels),1);

end