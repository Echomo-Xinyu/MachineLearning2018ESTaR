function [theta, J_history] = gradientDescentMulti(X, Y, theta, alpha, num_iters)
    
    % Initialize some useful values
    % The next line ignores the effect of
    m = length(Y(:, 1)); % number of training examples
    n = size(X, 2); % n-1 features, which means coefficients of X and one constant
    J_history = zeros(num_iters, 1);
    
    for iter = 1:num_iters 
        T = zeros(n,1);
        H = X * theta;
        for i=1:m,
            T = T + (H(i)-Y(i)) * X(i,:)';
        end;
    
        theta = theta - alpha * T / m;
    
        % ============================================================
    
        % Save the cost J in every iteration    
        J_history(iter) = computeCostMulti(X, Y, theta);
    
    end
    
end
    