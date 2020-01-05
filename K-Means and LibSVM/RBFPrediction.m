load('tstD.mat');
load('trD.mat');
load('trLbs.mat');
ImgIds = ml_load('../bigbangtheory_v3/test.mat', 'imIds');

cmd = ['-c ', num2str(8192), ' -g ', num2str(1)];
model = svmtrain(trLbs, trD', cmd);

[~, n] = size(tstD);
tstLbs = ones(n, 1);
[Prediction] = svmpredict(tstLbs, tstD', model, '');

Id = ImgIds';
T = table(Id, Prediction);
writetable(T, 'predTestLabel_1.csv');