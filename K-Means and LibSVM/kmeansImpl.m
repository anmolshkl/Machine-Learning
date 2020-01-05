function [centroids] = kmeansImpl(X, K)
    chooseFirstK = 0;
    numIterations = 20;
    centers = selectCenters(X, K, chooseFirstK);
    [cluster, centroids, sumOfSquares, iteration] = kmeans(X, K, centers, numIterations);   
    centroids = centroids';
end
function [centers] = selectCenters(X, K, chooseFirstK)
    
    [n, ~] = size(X);
    perm = randperm(n);
    if chooseFirstK == 1
        centers = X(1:K, :);
    else
        centers = X(perm(1:K), :);
    end
end

function [cluster, centers, sumOfSquares, iteration] = kmeans(X, K, centers, numIterations)
    
    [n, d] = size(X); 
    prevCluster = zeros(n,1);
    
    for iteration = 1:numIterations
        
        %fprintf("Iteration %d\n", iteration);
        %Step 1 - Cluster Assignment Step
        %disp("Cluster Assignment Step");
        cluster = zeros(n,1);
        sumOfSquares = 0;
        for i=1:n
            %fprintf("i=%d\n",i); 
            minmDist = intmax('int64');
            for k=1:K
                diff = X(i, :) - centers(k, :);
                %disp(diff);
                normDist = norm(diff);
                dist = normDist * normDist;
                %fprintf("k=%d dist = %f\n",k, dist);
                if dist < minmDist
                    minmDist = dist;
                    c = k;
                end
            end
            sumOfSquares = sumOfSquares + minmDist;
            cluster(i,:) = c;
        end

        if cluster == prevCluster
            return
        end

        %Step 2 - Move Centroid Step
        %disp("Move Centroid Step");
        centers = zeros(K, d);
        for k=1:K
            pointsInCluster = X(cluster == k, :);
            centers(k, :) = mean(pointsInCluster);
        end
        
        prevCluster = cluster;
        %disp(cluster)
    end
    
end