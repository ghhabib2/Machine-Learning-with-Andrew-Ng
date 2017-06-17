function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

# Calculate the Forwad propocgation in order to calucate the values for H_theta

% Calculate X'Theta for the first layer
A1 = [ones(m,1) X];
Z2 = A1 *Theta1';
% Calculate the sigmoid values for the hidden layer
A2 = [ones(size(Z2),1) sigmoid(Z2)];
% Calucate X'Theta for the second layer
Z3 = A2*Theta2';
% Calculate the sigmoid activation function for output layer
h = sigmoid(Z3);


% Calculate a matrix with 5000 columns and 10 rows in order to store the 
% the information of labels for each input separately and preapre every thing
% for +vector implementation.

tempYv = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

costVal=((-tempYv .* log(h)) - ((1-tempYv) .* log(1-h)));

% Calculate the final value using the sum fonction

J=(1/m) * sum(sum(costVal));
 
% part 2 Solution

% Convert each theta matrix to a Theta vector
tempTheta1=Theta1(:,2:end);
tempTheta2=Theta2(:,2:end);

% Now to the powering part
tempTheta1Pow2=tempTheta1.^2;
tempTheta2Pow2=tempTheta2.^2;

% Add the regulaization to the cost calculation
J=(1/m) * sum(sum(costVal)) + (lambda/(2*m)) * (sum(sum(tempTheta1Pow2)) + sum(sum(tempTheta2Pow2)));

 % Solution for part 3
 
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
 
for t = 1:m,

	tempH = h(t, :)';
	tempA1 = A1(t,:)';
	tempA2 = A2(t, :)';
	tempYvec = tempYv(t, :)';

	tempDelta3 = tempH - tempYvec;
	tempZ2 = [1; Theta1 * tempA1];
  tempDelta2 = Theta2' * tempDelta3 .* sigmoidGradient(tempZ2);

  delta1 = delta1 + tempDelta2(2:end) * tempA1';
  delta2 = delta2 + tempDelta3 * tempA2';
  
end;

Theta1_grad = (1 / m) * delta1;
Theta2_grad = (1 / m) * delta2;


% Calculate the requlaization for gradient

% solution for gradient regularization
GradientTheta1 = [ zeros(size(Theta1, 1), 1) tempTheta1 ];
GradientTheta2 = [ zeros(size(Theta2, 1), 1) tempTheta2 ];
Theta1_grad = (1 / m) * delta1 + (lambda / m) * GradientTheta1;
Theta2_grad = (1 / m) * delta2 + (lambda / m) * GradientTheta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
