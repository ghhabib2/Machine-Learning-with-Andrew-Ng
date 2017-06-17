function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Selected values of lambda (you should not change this)
range_vec = [0.01 0.03 0.1 0.3 1 3 10 30];

% You need to return these variables correctly.
error_val = 1;

% Temperory Variables for finding the best C and sigma
tempC=0;
tempSigma=0;

% Surffing the Range Vector for Assigning the values for C
for i=1:length(range_vec)
    % Assign C value for training
  tempC=range_vec(i);
  % Surffing the Range Vector for Assigning the values for Sigma
  for j=1:length(range_vec)
    % Assign the Sigma value for training
    tempSigma=range_vec(j);
    % train the SVM Model
    tempModel=svmTrain(X, y, tempC, @(x1, x2) gaussianKernel(x1, x2, tempSigma)); 
    % Predict the values of y based on the train models
    predictedY=svmPredict(tempModel, Xval);
    % Calculate the cost
    temp_error_val=mean(double(predictedY ~= yval));
    % Check if the error rate is smaller than corrent value and assign if the 
    % asnwer is yes.
    if(temp_error_val<error_val)
      error_val=temp_error_val;
      sigma=tempSigma;
      C=tempC;
    endif
  endfor
endfor


% =========================================================================

end
