X = load("../digit/digit.txt");
Y = load("../digit/labels.txt");

fixedK = 1;
numIterations = 20;
K = [2, 4, 6];
for i = 1:length(K)
    centers = getCenters(X, K(i), fixedK);
    [cluster, updatedCenters, sumOfSquares, ~] = kmeans(X, K(i), centers, numIterations);
    [p1, p2, p3] = pairCountingMeasures(Y, cluster);
    fprintf("K=%d sum Of Squares=%f p1=%f p2=%f p3=%f\n", K(i), sumOfSquares, p1, p2, p3);
end