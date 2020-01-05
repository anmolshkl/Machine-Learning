% Get the training Data
[trD, trLb, ~, ~, ~, ~] = HW4_Utils.getPosAndRandomNeg();

% Setting up the hyper-parameters
C = 10;
tol = 0.0001;

% Run SVM
[w, b, ~, ~, ~] = svm(trD, trLb, C, tol);

% Generate Result File
HW4_Utils.genRsltFile(w, b, "val", "rsltFile");

% Compute AP
[ap, ~, ~] = HW4_Utils.cmpAP("rsltFile", "val");
disp(ap)