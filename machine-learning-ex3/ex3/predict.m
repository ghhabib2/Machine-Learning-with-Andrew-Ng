function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Calculate X'Theta for the first layer
A1 = [ones(m,1) X];
Z2 = A1 *Theta1';
% Calculate the sigmoid values for the hidden layer
A2 = [ones(size(Z2),1) sigmoid(Z2)];
% Calucate X'Theta for the second layer
Z3 = A2*Theta2';
% Calculate the sigmoid activation function for output layer
A3 = sigmoid(Z3);


% Find the prediction values for each input of X
[maxvalues, index_max] = max(A3, [], 2);
% Calculate the prediction based on the index 
p = index_max;

% =========================================================================


end
