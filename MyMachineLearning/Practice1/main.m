% Machine Learning framework -- using Linear Regression with multiple variables
%
% As the datasets are not yet provided by mentors,
% % I spent like three hours repicking up python skills to scrap datasets from data.gov.sg but failed
% % and instead of a dataset, I got a serial of data which I have no clue the meaning of each data as it's simply data
% so rather than scrap data from the data.gov.sg, I turn to pretend there is a dataset provided
% and assume the data is in the format of location(latitude and longtidue), 7*7 sets of data in the past with respective timing
% More features like windspeed, the weather can be added, but to simplify the model while building the framework, I choose to ignore at this moment
% 


clear; close all; clc

% load data
fprintf('Loading all the data ...\n');
data = load('dataSet.txt');
X = data(:, 3:44);
Y = data(:, 45:51);
Z = data(:, 1:2);
m = length(Z(1, :));

fprintf('Program paused. Press Enter to continue.\n');
pause;


% Here I plan to draw a 3D graph to demonstrate the PM2.5 level across Singapore
plot3Dgaph(X, Y, Z);

% to simplify the data, scale the deatures and set them to zero mean and one std
[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% ============================================================

% Gradient Descent
fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.006;
num_iters = 500;

% Init Theta and Run Gradient Descent 
theta = zeros(43, 1);
[theta, J_history] = gradientDescentMulti(X, Y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');





