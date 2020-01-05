classdef HW5_BoW    
% Practical for Visual Bag-of-Words representation    
% Use SVM, instead of Least-Squares SVM as for MPP_BoW
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 18-Dec-2015
% Last modified: 16-Oct-2018    
    
    methods (Static)
        function main()
            scales = [8, 16, 32, 64];
            normH = 16;
            normW = 16;
            %bowCs = HW5_BoW.learnDictionary(scales, normH, normW);
            load('centroids.mat', 'bowCs');
            
            [trIds, trLbs] = ml_load('../bigbangtheory_v3/train.mat',  'imIds', 'lbs');             
            tstIds = ml_load('../bigbangtheory_v3/test.mat', 'imIds'); 
                        
            %trD  = HW5_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
            %tstD = HW5_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
            
            %save('trD.mat', 'trD');
            %save('tstD.mat', 'tstD');
            %save('trLbs.mat', 'trLbs');
            load('trD.mat', 'trD');
            load('tstD.mat', 'tstD');
            load('trLbs.mat', 'trLbs');

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Write code for training svm and prediction here            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Default value for C and Gamma
            model = svmtrain(trLbs,trD', '-v 5');
            save('model.mat', 'model');
            
            % Tuning for C and gamma using RBF kernel, 5-fold CV
            disp("Tuning C and gamma with RBF Kernel");
            bestcv = 0;
            for log2c = -2:12
                for log2g = -1:3
                    cmd = ['-q -v 5 -c ', 2^log2c, ' -g ', 2^log2g];
                    cv = svmtrain(trLbs,trD', sprintf('-q -v 5 -c %f -g %f', 2^log2c, 2^log2g));
                    if (cv >= bestcv)
                        bestcv = cv; 
                        bestc = 2^log2c; 
                        bestg = 2^log2g;
                    end
                 end
            end
            
            % Chi Square kernel using default values of C and gamma
            % As mentioned in assignment, to find a good default value of
            % gamme I have used the avg of chi-squared distance
            % between training data points
            gamma = HW5_BoW.getDefaultValueForGamma(trD');
            [trainK, testK] = exponentialKernel(trD', tstD', gamma);
            
            % Tuning gamma for chi-squared kernel
            disp("Tuning C and gamma with exponential chi squared Kernel");
            bestCV = 0;
            for log2c = -1:13
                for log2g = -1:3
                    [trainK, ~] = exponentialKernel(trD', tstD', 2^log2g);
                    cmd = ['-q -t 4 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
                    cv = svmtrain(trLbs,trainK, cmd);
                    if (cv >= bestCV)
                        bestCV = cv; 
                        bestc = 2^log2c; bestg = 2^log2g;
                    end
                    fprintf('(CHI, currentc=%g, currentg=%g, currentrate=%g, bestc=%g, bestg=%g, bestrate=%g)\n',2^log2c, 2^log2g, cv, bestc, bestg, bestcv);
                 end
            end
        end
        
        function [gamma] = getDefaultValueForGamma(trnD)
            n = size(trnD, 0);
            summation = 0;
            count = 0;
            for i = 1:n
                for j = i+1:n
                    count = count + 1;
                    [distance] = HW5_BoW.getChiSquareDistance(trnD(i,:)', trnD(j, :)');
                    summation = summation + distance;
                end
            end

            gamma = summation / count;
        end

        function [dist] = findChiSquareDistance(x, y)
            numerator = (x - y).^2;
            denominator = x + y + 0.00005;
            dist = sum(numerator./denominator);
        end
                
        function bowCs = learnDictionary(scales, normH, normW)
            % Number of random patches to build a visual dictionary
            % Should be around 1 million for a robust result
            % We set to a small number her to speed up the process. 
            nPatch2Sample = 100000;
            
            % load train ids
            trIds = ml_load('../bigbangtheory_v3/train.mat', 'imIds'); 
            nPatchPerImScale = ceil(nPatch2Sample/length(trIds)/length(scales));
                        
            randWins = cell(length(scales), length(trIds)); % to store random patches
            for i=1:length(trIds);
                ml_progressBar(i, length(trIds), 'Randomly sample image patches');
                im = imread(sprintf('../bigbangtheory_v3/%06d.jpg', trIds(i)));
                im = double(rgb2gray(im));  
                for j=1:length(scales)
                    scale = scales(j);
                    winSz = [scale, scale];
                    stepSz = winSz/2; % stepSz is set to half the window size here. 
                    
                    % ML_SlideWin is a class for efficient sliding window 
                    swObj = ML_SlideWin(im, winSz, stepSz);
                    
                    % Randomly sample some patches
                    randWins_ji = swObj.getRandomSamples(nPatchPerImScale);
                    
                    % resize all the patches to have a standard size
                    randWins_ji = reshape(randWins_ji, [scale, scale, size(randWins_ji,2)]);                    
                    randWins{j,i} = imresize(randWins_ji, [normH, normW]);
                end
            end
            randWins = cat(3, randWins{:});
            randWins = reshape(randWins, [normH*normW, size(randWins,3)]);
                                    
            fprintf('Learn a visual dictionary using k-means\n');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Use your K-means implementation here                       %
            % to learn visual vocabulary                                 %
            % Input: randWinds contains your data points                 %
            % Output: bowCs: centroids from k-means, one column for each centroid  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            K=1000;
            [bowCs] = kmeansImpl(randWins', K);
            save('centroids.mat', 'bowCs');
        end
                
        function D = cmpFeatVecs(imIds, scales, normH, normW, bowCs)
            n = length(imIds);
            D = cell(1, n);
            startT = tic;
            for i=1:n
                ml_progressBar(i, n, 'Computing feature vectors', startT);
                im = imread(sprintf('../bigbangtheory_v3/%06d.jpg', imIds(i)));                                
                bowIds = HW5_BoW.cmpBowIds(im, scales, normH, normW, bowCs);                
                feat = hist(bowIds, 1:size(bowCs,2));
                D{i} = feat(:);
            end
            D = cat(2, D{:});
            D = double(D);
            D = D./repmat(sum(D,1), size(D,1),1);
        end        
        
        % bowCs: d*k matrix, with d = normH*normW, k: number of clusters
        % scales: sizes to densely extract the patches. 
        % normH, normW: normalized height and width oMf patches
        function bowIds = cmpBowIds(im, scales, normH, normW, bowCs)
            im = double(rgb2gray(im));
            bowIds = cell(length(scales),1);
            for j=1:length(scales)
                scale = scales(j);
                winSz = [scale, scale];
                stepSz = winSz/2; % stepSz is set to half the window size here.
                
                % ML_SlideWin is a class for efficient sliding window
                swObj = ML_SlideWin(im, winSz, stepSz);
                nBatch = swObj.getNBatch();
                
                for u=1:nBatch
                    wins = swObj.getBatch(u);
                    
                    % resize all the patches to have a standard size
                    wins = reshape(wins, [scale, scale, size(wins,2)]);                    
                    wins = imresize(wins, [normH, normW]);
                    wins = reshape(wins, [normH*normW, size(wins,3)]);
                    
                    % Get squared distance between windows and centroids
                    dist2 = ml_sqrDist(bowCs, wins); % dist2: k*n matrix, 
                    
                    % bowId: is the index of cluster closest to a patch
                    [~, bowIds{j,u}] = min(dist2, [], 1);                     
                end                
            end
            bowIds = cat(2, bowIds{:});
        end        
        
    end    
end

