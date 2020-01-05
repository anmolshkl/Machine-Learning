%3.4.4 - Training on exponential Chi square kernel using default value of C
%and gamma %

load('tstD.mat');
load('trD.mat');
load('trLbs.mat');

[gamma] = getGammaDefault(trD');
fprintf("Gamma = %g\n", gamma);
[trainK] = cmpExpX2Kernel(trD', tstD', gamma);


%3.4.5 - Tuning C and gamma for SVM with exponential chi square kernel using 5-fold cross validation
disp("Tuning C and gamma with exponential chi square Kernel");
bestcv = 0;
for log2c = -1:13
    for log2g = -1:4
        [trainK] = cmpExpX2Kernel(trD', tstD', 2^log2g);
        cmd = ['-q -t 4 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(trLbs,trainK, cmd);
        if (cv >= bestcv)
            bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('(CHI, currentc=%g, currentg=%g, currentrate=%g, bestc=%g, bestg=%g, bestrate=%g)\n',2^log2c, 2^log2g, cv, bestc, bestg, bestcv);
     end
end

function [gamma] = getGammaDefault(trainD)
    %% Get all pairs in training data %%
    [n, ~] = size(trainD);
    summ = 0;
    count = 0;
    for i=1:n
        for j=i+1:n
            [dist] = HW5_BoW.getChiSquareDistance(trainD(i,:)', trainD(j, :)');
            summ = summ + dist;
            count = count + 1;
        end
    end

    gamma = summ/count;
end

function [dist] = getChiSquareDistance(x, y)
    epsilon = 0.00001;
    nume = (x - y).^2;
    deno = x + y + epsilon;

    fraction = nume./deno;

    dist = sum(fraction);
end

