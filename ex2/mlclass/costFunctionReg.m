function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y) % number of training examples
n = size(X)(2)

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

function hyp = hyp(index)
	hyp = sigmoid(X(index, :)*theta);
endfunction

function cost = computeCost
	cost = 0.0;
	regFactor = 0.0;
	for index = [1:m]
		cost += -y(index)*log(hyp(index)) - (1-y(index))*log(1-(hyp(index)));
	endfor
	cost *= (1/m);
	for index = [1:n]
		if (index > 1)
			regFactor += theta(index)^2;
		endif
	endfor
	regFactor *= (lambda/(2*m));
	cost += regFactor;
endfunction

function derivatives = computeDerivatives
	derivatives = zeros(size(theta)(1), size(theta)(2));
	for jay = [1:size(derivatives)(1)]
		for index = [1:m]
			singularCost = (hyp(index) - y(index));
			derivatives(jay) += singularCost * X(index, jay);
		endfor
		derivatives(jay) /= m;
		if (jay > 1)
			derivatives(jay) += (lambda/m * theta(jay));
		endif
	endfor
endfunction

J = computeCost;
grad = computeDerivatives;

% =============================================================

end
