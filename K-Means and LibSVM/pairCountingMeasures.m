function [p1, p2, p3] = pairCountingMeasures(Y, cluster)
    [n, ~] = size(Y);
    sameLabel = 0;
    diffLabel = 0;
    sameCluster = 0;
    diffCluster = 0;
 
    for i=1:n
        for j=i+1:n
            if Y(i,:) == Y(j,:)
                sameLabel = sameLabel + 1;
                if cluster(i, :) == cluster(j, :)
                    sameCluster = sameCluster + 1;
                end
            else
                diffLabel = diffLabel + 1;
                if cluster(i, :) ~= cluster(j, :)
                    diffCluster = diffCluster + 1;
                end
            end
         end
    end
    p1 = (sameCluster * 100)/sameLabel;
    p2 = (diffCluster * 100)/diffLabel;
    p3 = (p1+p2)/2;                                     
end
