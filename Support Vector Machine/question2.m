% This file includes the following questions - 
% Question 2.1
% Question 2.2 
% Question 2.3
% Question 2.4
% Question 2.5 

data = load('q2_1_data.mat');

X_train = data.trD;
y_train = data.trLb;

X_val = data.valD;
y_val = data.valLb;

C = [0.1, 10];
tol = 0.00001;

% Loop over all the Cs
for c = C
    fprintf("----------Training on C = [%f]------------\n", c);
    
    [w, b, obj, nSV, ~] = svm(X_train, y_train, c, tol);
    
    generateStats(X_train, y_train, w, b, obj, nSV, "Training");
    generateStats(X_val, y_val, w, b, obj, nSV, "Validation");
end

function [] = generateStats(X, y, w, b, obj, nSV, mode)
    y_pred = predict(X, w, b);
    accuracy = calculateAccuracy(y, y_pred);
    if mode == "Training"
        fprintf("%s accuracy: [%f]\n", mode, accuracy);
    else
        fprintf("%s accuracy: [%f] \nObjective Value: [%f] \nSupport Vectors: [%d]\n",mode, accuracy, obj, nSV);
    end
    fprintf("%s Confusion Matrix -\n", mode);
    disp(confusionmat(y, y_pred));
end
function [accuracy] = calculateAccuracy(y, y_pred)
    n = numel(y);
    accuracy = 100 * (sum(y == y_pred) / n);
end

function [y_pred] = predict(X, w, b)
    y_pred = (X' * w) + b;
    y_pred(y_pred >= 0) = 1;
    y_pred(y_pred < 0) = -1;
end