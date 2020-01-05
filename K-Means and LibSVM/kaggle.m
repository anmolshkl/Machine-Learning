load('tstD.mat');
load('trD.mat');
load('trLbs.mat');

[trainK, testK] = exponentialKernel(trD', tstD', 2);
cmd = ['-t 4 -c ', num2str(4196), ' -g ', num2str(0.5)];
model = svmtrain(trLbs, trainK, cmd);

[n, ~] = size(testK);
tstLbs = ones(n, 1);
[Prediction] = svmpredict(tstLbs, testK, model);

Id = 1:1600;
Id = Id';
T = table(Id, Prediction);
writetable(T, 'submission.csv');