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

%% Matrix a2 caculation (hidden layer):

% vector x = a1 (400, 1) => maxtrix X (5000, 400)
% add column x(0) = 1  => X (5000, 401)
a1 = [ones(m,1) X];

% Theta1 (a2, x+1) = (25, 401) 
% vector a2 (25, 1) => Matrix a2 (5000, 25)
a2 = sigmoid(a1*Theta1');

%% Matrix h(x) caculation (output layer):

% add column a2(0) = 1 => Now a2 (5000, 26)
a2_height = size(a2,1);
a2 = [ones(a2_height, 1) a2];

% a3 = h(x)
% Theta2 (a3, a2 +1) = (10, 26)
% vector a3 (10, 1) => Matrix a3 (5000, 10)
a3 = sigmoid(a2*Theta2');

%% Matrix Y 
% each y (5000, 1) has value from 1 to 10 (10 = 0)
% need to turn that value to vector
% example 4 = [0;0;0;1;0;0;0;0;0;0]
Y = zeros(m, num_labels);

for i = 1:m
    class = y(i);
    Y(i,class) = 1;
end
% Y now should be (5000, 10)

%% Matrix all_J to hold all the costs
% all_J should be (5000, 10)

all_J = (Y.*log(a3) + (1 - Y).*log(1-a3));

J = sum(all_J(:));
J = - J / m;

%% Regularized cost function

% Remove theta(0) in each theta, call it heta
Heta1 = Theta1;
Heta1(:,1) = [];

Heta2 = Theta2;
Heta2(:,1) = [];

% Square each Heta element then add them together
sum_heta = sum(Heta1(:).^2) + sum(Heta2(:).^2);

% Cost function regularized
J = J + sum_heta * lambda /(2 * m);

%% Backpropagation

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for i = 1:m
    
    % Initialize a1, a2 with bias unit
    a1 = [1 ; X(i,:)'];             % size (401, 1)
    a2 = [1; sigmoid(Theta1*a1)];   % size (26, 1)
    a3 = sigmoid(Theta2*a2);        % size (10, 1)
    
    % Compute delta
    delta3 = a3 - Y(i,:)';    % size (10, 1)
    delta2 = (Theta2' * delta3) .* a2 .* (1 - a2);  % size (26, 1)
    % remove bias unit in delta2
    delta2 = delta2(2:end);
    
    % Big delta (same size as Theta1 & 2)
    Delta2 = Delta2 + delta3 * a2'; 
    Delta1 = Delta1 + delta2 * a1';
   
    % regularize unit
    reg1 = lambda * Theta1;
    reg1(:,1) = 0;
    
    reg2 = lambda * Theta2;
    reg2(:,1) = 0;
    
    Theta1_grad = (Delta1 + reg1) / m;
    Theta2_grad = (Delta2 + reg2) / m;
end

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
