% Load the ground truth data
data = load('clock_2hzrn.mat', 'clock_2hzrn');
stopSignsAndCars = data.clock_2hzrn;

% Display a summary of the ground truth data
summary(clock_2hzrn)

% Only keep the image file names and the stop sign ROI labels
stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});

% Display one training image and the ground truth bounding boxes
I = imread(stopSigns.imageFilename{1});
I = insertObjectAnnotation(I,'Rectangle',stopSigns.stopSign{1},'stop sign','LineWidth',8);

figure
imshow(I)