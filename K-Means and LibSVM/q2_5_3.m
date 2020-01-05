X = load("../digit/digit.txt");
Y = load("../digit/labels.txt");

fixedK = 0;
iterations = 20;
sosList = [];
kList = [1, 2, 3];
p1List = []; p2List = []; p3List = [];
iterationCount = 1000;
for K = 1:10
    sumP1 = 0; sumP2 = 0; sumP3 = 0; sos = 0;
    for iter = 1:iterationCount
        centers = getCenters(X, K, fixedK);
        [cluster, updatedCenters, sumOfSquares, iteration] = kmeans(X, K, centers, iterations);
        [p1, p2, p3] = pairCountingMeasures(Y, cluster);
        sos = sos + sumOfSquares;
        sumP1 = sumP1 + p1;
        sumP2 = sumP2 + p2;
        sumP3 = sumP3 + p3;
    end

    avgP1 = sumP1/iterationCount;
    avgP2 = sumP2/iterationCount;
    avgP3 = sumP3/iterationCount;
    sos = sos/iterationCount;

    fprintf("K=%d sumOfSquares=%f p1=%f p2=%f p3=%f\n",K, sos, avgP1, avgP2, avgP3);
    sosList = [sosList, sos];
    p1List = [p1List, avgP1];
    p2List = [p2List, avgP2];
    p3List = [p3List, avgP3];
end

plotGraph(kList, sosList, p1List, p2List, p3List);