function [X_norm, mu, sigma] = featureNormalize(X)
% This is to return a set of normalized X values
% which the mean value(mu) is 0 and the standard derivative is 1

% initialization
X_norm = X;
mu = zeros(1, size(X, 44));
sigma = zeros(1, size(X, 44));

% the original mean value
mu = mean(X);

% make the mean value be 0
for k=1:length(X(:,1)),
    X_norm(k,:) = X_norm(k,:) - mu;
end;

% make the standard derivative be 1
for i=1:44,
    % the original standard derivative
    sigma(:, i) = std(X(:, i));
    X_norm(:, i) = X_norm(:, i) ./ sigma(:, i)
end;    



end