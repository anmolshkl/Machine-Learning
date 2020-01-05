X_train = readtable('Train_Features.csv');
y_train = readtable('Train_Labels.csv', 'HeaderLines',1);
X_val = readtable('Val_Features.csv');
y_val = readtable('Val_Labels.csv', 'HeaderLines',1);
X_test = readtable('Test_Features.csv');

X_train(:,1) = [];
y_train(:,1) = [];
X_val(:,1) = [];
y_val(:,1) = [];
id = X_test(:, 1);
X_test(:,1) = [];

% Convert table to array
X_train = double(table2array(X_train));
X_val = double(table2array(X_val));
X_test = double(table2array(X_test));
y_train = double(table2array(y_train));
y_val = double(table2array(y_val));
id = table2array(id);

X_train = X_train';
X_test = X_test';
X_val = X_val';
X_t = [X_train X_val];
y_t = [y_train; y_val];
% X_t = normalize(X_t, 2, 'zscore');
% X_val = normalize(X_val, 2, 'zscore');
% X_test = normalize(X_test, 2, 'zscore');
C = 0.000556;
tol = 0.000001;
uniqueLb = 4;

d = size(X_t, 1);
n = size(X_t, 2);

wMat = zeros(d, uniqueLb);
bMat = zeros(1, uniqueLb);

for i = 1:uniqueLb
    fprintf("----------Training classifier for class = [%d]------------\n", i);

    % Forming new Y for training
    y = zeros(n, 1);
    y(y_t ~= i) = -1;
    y(y_t == i) = 1;

    % Training SVM
    [w, b, ~, ~, ~] = svm(X_t, y, C, tol);
    wMat(:,i) = w;
    bMat(:,i) = b;
end
    
y_pred = getPrediction(X_t, wMat, bMat);
accuracy_train = getAccuracy(y_t, y_pred);
fprintf("Training accuracy: [%f]\n", accuracy_train);

y_val_pred = getPrediction(X_val, wMat, bMat);
accuracy_val = getAccuracy(y_val, y_val_pred);
fprintf("Validation accuracy: [%f]\n", accuracy_val);

y_test_pred = getPrediction(X_test, wMat, bMat);
writecell([id, num2cell(y_test_pred)], "submission.csv");

function [y_pred] = getPrediction(X, wMat, bMat)
    n = size(X, 2);
    uniqueLb = size(wMat,2);

    confidenceMatrix = zeros(n, uniqueLb);
    
    for i = 1 : uniqueLb
        w = wMat(:,i);
        b = bMat(:,i);
        
        confidenceMatrix(:,i) = ((X' * w) + b);
    end
    
    [~, y_pred] = max(confidenceMatrix, [], 2);
end

function [accuracy] = getAccuracy(y, y_pred)
    n = numel(y);
    accuracy = 100 * (sum(y == y_pred) / n);
end