X = load("../digit/digit.txt");
Y = load("../digit/labels.txt");

fixedK = 1;
iterations = 20;
K = 6;
centers = getCenters(X, K, fixedK);
[cluster, updatedCenters, sumOfSquares, iteration] = kmeans(X, K, centers, iterations);
[p1, p2, p3] = pairCountingMeasures(Y, cluster);
fprintf("K=%d iterations=%d sum of Squares=%f p1=%f p2=%f p3=%f\n",K,iteration, sumOfSquares, p1, p2, p3);
