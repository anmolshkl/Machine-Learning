function [centers] = getCenters(X, K, fixedK)
    [n, ~] = size(X);
    perm = randperm(n);
    if fixedK == 1
        centers = X(1:K, :);
    else
        centers = X(perm(1:K), :);
    end
end