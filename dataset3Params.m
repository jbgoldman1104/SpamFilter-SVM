function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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

step = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
predErr = zeros(length(step));

% calculate the prediction error of all combinations of C and sigma
for i = 1 : length(step)
    for j = 1 : length(step)
        C = step(i);
        sigma = step(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        predErr(i, j) = mean(double(predictions ~= yval));
    end
end

% find the index of the pair where minimum prediction error occurs
minimum = min(min(predErr));
[indexC, indexSigma] = find(predErr == minimum);
C = step(indexC);
sigma = step(indexSigma);


% =========================================================================

end
