% This file includes the following questions - 
% Question 3.4.2
% Question 3.4.3
% Question 3.4.4

[trainData, trainLabel, valData, valLabel, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();

trainData = [trainData valData];
trainLabel = [trainLabel; valLabel];

PosD = trainData(:, trainLabel == 1);
PosLb = trainLabel(trainLabel == 1);

NegD = trainData(:, trainLabel == -1);
NegLb = trainLabel(trainLabel == -1);

m = 1000;
C = 10;
tol = 0.0001;
maxIterations = 20;

apMat = zeros(maxIterations, 1);
objMat = zeros(maxIterations, 1);

load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, "train"), 'ubAnno');

[w, b, ~, ~, alpha] = svm(trainData, trainLabel, C, tol);


for i = 1 : maxIterations
    [trainData, trainLabel] = deleteAllNonSVInNegativeDataset(trainData, trainLabel, alpha, tol);
    numImagesInDirectory = 93;
    newNegD = findHardestNegativeExamples(w, b, ubAnno, numImagesInDirectory);
    trainData = [trainData, newNegD];
    trainLabel = [trainLabel; -1 * ones(size(newNegD, 2), 1)];
    
    [w, b, obj, ~, alpha] = svm(trainData, trainLabel, C, tol);
    [ap, ~, ~] = HW4_Utils.cmpAP("mining_val", "val");
    HW4_Utils.genRsltFile(w, b, "val", "mining_val");
    apMat(i) = ap;
    objMat(i) = obj;
end

HW4_Utils.genRsltFile(w, b, "test", "112551470");
disp("Question 3.4.3:");
disp("------------------");
disp("The Objective Values are:");
disp(objMat);

disp("Average Precision Values:");
disp(apMat);

iterations = 1 : maxIterations;
iterations = iterations(:);

figure
plot(iterations, objMat);
title('Objective Values Plot');
xlabel('Iteration');
ylabel('Objective Values');

figure
plot(iterations, apMat);
title('APs Plot');
xlabel('Iteration');
ylabel('APs');

function[trainData, trainLabel] = deleteAllNonSVInNegativeDataset(trainData, trainLabel, alpha, tol)
    % Removing Negative Non Support vectors from the data.
    idx = find((alpha < tol) & (trainLabel == -1));
    trainData(:, idx) = [];
    trainLabel(idx, :) = [];
end

function[newNegativeData] = findHardestNegativeExamples(w, b, ubAnno, numImagesInDirectory)
    overlapThreshold = 0.1;
    newNegativeData = [];
    m = 1000;
    for j = 1 : numImagesInDirectory
        im = sprintf('%s/trainIms/%04d.jpg', HW4_Utils.dataDir, j);
        im = imread(im);
        
        rect = HW4_Utils.detect(im, w, b);
        total_pos = sum(rect(end,:)>0);
        negRectangle = rect(:, 1:total_pos + 10);
                
        [imH, imW, ~] = size(im);
        negRectangle = negRectangle(:, negRectangle(3, :) < imW);
        negRectangle = negRectangle(:, negRectangle(4, :) < imH);
        
        negRectangle = negRectangle(1:4, :);
        ubs = ubAnno{j};
        
        for k = 1 : size(ubs, 2)
            overlap = HW4_Utils.rectOverlap(negRectangle, ubs(:, k));                    
            negRectangle = negRectangle(:, overlap < overlapThreshold);
        end
        
        newNegativeData = convertToGreyScale(negRectangle, im);
        
        if size(newNegativeData, 2) > m
            break;
        end
    end
end

function[newNegativeData] = convertToGreyScale(negRectangle, im)
    newNegativeData = [];
    % Convert the images to greyscale and normalize them
    for k = 1 : size(negRectangle, 2)
        tmp = negRectangle(:, k);
        imReg = im(tmp(2):tmp(4), tmp(1):tmp(3),:);
        imReg = imresize(imReg, HW4_Utils.normImSz);

        feature = HW4_Utils.cmpFeat(rgb2gray(imReg));
        feature = HW4_Utils.l2Norm(feature);
        newNegativeData = [newNegativeData, feature];
    end
end