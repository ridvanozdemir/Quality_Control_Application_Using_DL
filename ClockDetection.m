%% Object Detection Using Faster R-CNN Deep Learning
% This example shows how to train an object detector using a deep learning
% technique named Faster R-CNN (Regions with Convolutional Neural
% Networks).
%
% Copyright 2017 The MathWorks, Inc.
%% note if you do not want to open the labeler app

data =load('clock_datastore.mat', 'C');
C = data.C;    
summary(C)

% Display first few rows of the data set.
C(1:4,:)

%% Read one of the images.
I = imread(C.Var1{10});

% Insert the ROI labels.
I2 = insertObjectAnnotation(I, 'Rectangle', C.clock{10}, 'clock');

% Resize and display image.
figure;
imshowpair(imresize(I,3), imresize(I2, 3), 'montage');


%% Split data into a training and test set.

idx = floor(0.1 * height(C));
trainingData = C(1:idx,:);
testData = C(idx:end,:);

%% Create a Convolutional Neural Network (CNN)
% Either explain creating the CNN from scratch and training options
[layers,trainingOpts] = createCNN(width(C));

%% Train Faster R-CNN
doTrainingAndEval = false;

if doTrainingAndEval
    % Set random seed to ensure example training reproducibility.
    rng(0);
    
    % Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
    % for finer resolution for multiscale object detection.
    %detector = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
    detector = trainFasterRCNNObjectDetector(trainingData, layers, trainingOpts, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'BoxPyramidScale', 1.2);
else
    % Load pretrained detector for the example.
    data = load('fasterRCNNVehicleTrainingData.mat');
    
    detector = data.detector;
end
%% Evaluate Detector Using Test Set

if doTrainingAndEval
    % Run detector on each image in the test set and collect results.
    resultsStruct = struct([]);
    for i = 1:height(testData)
        
        % Read the image.
        I = imread(testData.Var1{i});
        
        % Run the detector.
        [bboxes, scores, labels] = detect(detector, I);
        
        % Collect the results.
        resultsStruct(i).Boxes = bboxes;
        resultsStruct(i).Scores = scores;
        resultsStruct(i).Labels = labels;
    end
    
    % Convert the results into a table.
    results = struct2table(resultsStruct);
else
    % Load results from disk.
    results = data.results;
end

% Extract expected bounding box locations from test data.
expectedResults = testData(:, 2:end);

% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

%% Plot precision/recall curve
figure
plot(recall, precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.1f', ap))
