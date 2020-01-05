function [cluster, centers, sumOfSquares, iteration] = kmeans(X, K, centers, iterations)
    
    [n, d] = size(X); 
    previousCluster = zeros(n,1);
    
    for iteration = 1:iterations
        
        cluster = zeros(n,1);
        sumOfSquares = 0;
        for i=1:n
            minmDist = intmax('int64');
            for k=1:K
                diff = X(i, :) - centers(k, :);
                normDist = norm(diff);
                dist = normDist * normDist;
                if dist < minmDist
                    minmDist = dist;
                    c = k;
                end
            end
            cluster(i,:) = c;
            sumOfSquares = sumOfSquares + minmDist;
        end

        if cluster == previousCluster
            return
        end
        centers = zeros(K, d);
        for k=1:K
            pointsInCluster = X(cluster == k, :);
            centers(k, :) = mean(pointsInCluster);
        end
        previousCluster = cluster;

    end
    
end

