load('trD.mat')
[trIds, trLbs] = ml_load('../bigbangtheory_v2/train.mat',  'imIds', 'lbs');
bestcv = 0;
for log2c = 10:15
  for log2g = -1:8
    cmd = ['-q -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(trLbs, trD', cmd);
    if (cv >= bestcv)
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end