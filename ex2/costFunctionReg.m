function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% h of x
h = sigmoid(X * theta);

% heta is theta without first element
heta = theta(2:size(theta));

% regularize part
reg = lambda * (2*m)^-1 * (heta' * heta) ;

% cost function and gradient
J = - 1/m * (y' * log(h) + (1 - y)' * log(1 - h)) + reg;

grad(1) = 1/m * sum(h - y); % X at column 1 is ones, replace with sum
grad(2:size(grad)) = 1/m * X(:,2:size(X,2))' * (h - y) + heta * lambda / m;
%                          ^ X from column 2 to end

% =============================================================

end
