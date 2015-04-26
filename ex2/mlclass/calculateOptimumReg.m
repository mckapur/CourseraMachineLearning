function [theta, cost] = calculateOptimumReg(theta, X, y, lambda)

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), theta, options);

end