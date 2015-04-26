function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

function hyp = hyp(index)
	hyp = sigmoid(X(index, :)*theta);
endfunction

function cost = computeCost
	cost = 0.0;
	for index = [1:m]
		cost += -y(index)*log(hyp(index)) - (1-y(index))*log(1-(hyp(index)));
	endfor
endfunction

function derivatives = computeDerivatives
	derivatives = zeros(size(theta)(1), size(theta)(2));
	for jay = [1:size(derivatives)(1)]
		for index = [1:m]
			singularCost = (hyp(index) - y(index));
			derivatives(jay) += singularCost * X(index, jay);
		endfor
	endfor
	derivatives /= m;
endfunction

J = (1/m) * computeCost;
grad = computeDerivatives;

% =============================================================

end
