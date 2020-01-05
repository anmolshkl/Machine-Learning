function [w, b, obj, nSV, alpha] = svm(X, y, C, tol)

    d = size(X, 1);
    n = size(X, 2);

    X = double(X);
    y = double(y);
    
    % Using linear kernal
    kern = @(X,d) (X);

    K = kern(X' * X, 6);	
   
    H = K .* (y * y');
    f = -1 * ones(n, 1);
    A = [];
    b = [];
    Aeq = y';
    beq = zeros(1, 1);
    lb = zeros(n, 1);
    ub = C * ones(n, 1);

    [alpha, fval] = quadprog(H, f, A, b, Aeq, beq, lb, ub);
    
    nSV = size(find(alpha > tol), 1);
    
    w = (diag(alpha) * y)' * X';
    w = w';
    
    % Calculating bias
    b = y - (X'*w);
    b = mean(b);
    
    % Calculating Objective value
    temp = sum(((alpha .* y) .* X'), 1);
    obj = sum(alpha) - (0.5 * norm(temp) * norm(temp));
%     obj = fval;
end