[trainData, trainLabel, valData, valLabel, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();

trainData = [trainData valData];
trainLabel = [trainLabel; valLabel];

PosD = trainData(:,trainLabel == 1);
PosLb = trainLabel(trainLabel == 1);

NegD = trainData(:,trainLabel == -1);
NegLb = trainLabel(trainLabel == -1);

m = 1000;
C = 10;
tol = 0.00001;
overlapThreshold = 0.1;
maxIterations = 10;

load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, "val"), 'ubAnno');
ubAnnoVal = ubAnno;
load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, "train"), 'ubAnno');

[w, b, ~, ~, alpha] = svm(trainData, trainLabel, C, tol);
for i = 1 : maxIterations
    % Removing Negative Non Support vectors from the data.
    idx = find((alpha < tol) & (trainLabel == -1));
    trainData(:, idx) = [];
    trainLabel(idx, :) = [];
    
    newNegD = [];
    for j = 1 : 185
        if j <= 93
            im = sprintf('%s/trainIms/%04d.jpg', HW4_Utils.dataDir, j);
            ubs = ubAnno{j};
        else
            im = sprintf('%s/trainIms/%04d.jpg', HW4_Utils.dataDir, j-93);
            ubs = ubAnnoVal{j-93};
        end    
        im = imread(im);
        
        rect = HW4_Utils.detect(im, w, b);
        %rectNeg = rect(:, (rect(5,:) < 0));
        
        total_pos = sum(rect(end,:)>0);
        rectNeg = rect(:, 1:total_pos + 10);
                
        % Prune rectangles that are not within image boundaries
        [imH, imW, ~] = size(im);
        rectNeg = rectNeg(:, rectNeg(3, :) < imW);
        rectNeg = rectNeg(:, rectNeg(4, :) < imH);
        
        % Prune rectangles where overlap is more than threshold
        rectNeg = rectNeg(1:4, :);
        
        for k = 1 : size(ubs, 2)
            overlap = HW4_Utils.rectOverlap(rectNeg, ubs(:, k));                    
            rectNeg = rectNeg(:, overlap < overlapThreshold);
        end
        
        % Feature extraction
        for k = 1 : size(rectNeg, 2)
            tmp = rectNeg(:, k);
            imReg = im(tmp(2):tmp(4), tmp(1):tmp(3),:);
            imReg = imresize(imReg, HW4_Utils.normImSz);
            
            feature = HW4_Utils.cmpFeat(rgb2gray(imReg));
            feature = HW4_Utils.l2Norm(feature);
            newNegD = [newNegD, feature];
        end
        
        if size(newNegD, 2) > m
            break;
        end
    end
    
    trainData = [trainData, newNegD];
    trainLabel = [trainLabel; -1 * ones(size(newNegD, 2), 1)];
    
    [w, b, ~, ~, alpha] = svm(trainData, trainLabel, C, tol);
end

HW4_Utils.genRsltFile(w, b, "test", "112551470");