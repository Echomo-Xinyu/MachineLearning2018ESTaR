% I planned to plot data between two variables
% but obviously I can't put both time, the PM2.5 and the location in the same 2D graph
% So now this is a tryout for 3D graph

function plot3DGraph = myFun(X, Y, Z)
%myFun - Description
%
% Syntax: plot3DGraph = myFun(input)
%
% Long description
    
for i=1:42
    % location as x and y axes, PM2.5 level as 
    x_intercept = Z(1, i);
    y_intercept = Z(2, i);

    if i <= 42
        z_intercept = X(:, i);
        plot(x_intercept, y_intercept, z_intercept);
    else
        z_intercept = Y(:, i);
        plot(x_intercept, y_intercept, z_intercept);
    end

    fprintf('Press Enter to continue to the graph of next moment\n');
    pause;
end    

end