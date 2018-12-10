function [J, grad] = costFunction(theta, X, y)
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta)); %creates a vector with temporary values for all values of theta.
h = sigmoid(X*theta); %calls sigmoid function and multiplied it by input theta to find hypothesis
% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h)); %first equation for J
J = (1/m)*(-y'* log(h) - (1 - y)'* log(1-h)); %second equation for J
grad = (1/m)*X'*(h - y); %update theta values based on equation
end
