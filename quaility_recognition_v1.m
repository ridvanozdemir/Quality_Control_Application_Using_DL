%% Load Training Images 
allImages = imageDatastore('hatali_hatasiz_c', 'IncludeSubfolders', true,'LabelSource', 'foldernames');

%% Split data into training and test sets 
%[trainingImages, testImages] = splitEachLabel(allImages, 0.9, 'randomize'); 
[trainingImages, validationImages] = splitEachLabel(allImages, 0.9, 'randomize'); 

trainingImages.countEachLabel
%% Load Pre-trained Network (AlexNet) 
alex = alexnet;

%% Review Network Architecture 
layers = alex.Layers;
layers
%% Modify Pre-trained Network 
numClasses = numel(categories(trainingImages.Labels)); 

layers(23) = fullyConnectedLayer(numClasses); % change this based on # of classes 
layers(25) = classificationLayer

%% Perform Transfer Learning 
% mini batch 16, validation 30, acc 97 
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001,... 
'MaxEpochs', 6, ...
'MiniBatchSize', 16, ...
'Shuffle','every-epoch', ...
'ValidationData',validationImages, ...
'ValidationFrequency',5, ...
'ValidationPatience',Inf, ...
'Verbose',true, ... 
'Plots','training-progress');

%% Set custom read function 

trainingImages.ReadFcn = @readFunctionTrain;

%% Train the Network 

QNet = trainNetwork(trainingImages, layers, opts);

%% Test Network Performance 

validationImages.ReadFcn = @readFunctionTrain; 
predictedLabels = classify(QNet, validationImages); 
accuracy = mean(predictedLabels == validationImages.Labels)

% Somewhat easy confusion matrix - heat map (Base MATLAB - new in 17A)
tt = table(validationImages.Labels,predictedLabels,'VariableNames',{'Actual','Predicted'});
figure; heatmap(tt,'Predicted','Actual');


% Choose a random image, visualize the results and show the confidence
randNum = randi(length(validationImages.Files));
im_display = imread(validationImages.Files{randNum});

imt = readFunctionTrain(validationImages.Files{randNum}) ;
[label,scr] = classify(QNet,imt); % classify with deep learning 
imshow(im_display);
title(sprintf('%s %.2f', char(label),max(scr)));