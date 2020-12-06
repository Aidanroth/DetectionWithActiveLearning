%Set up data and network
net = resnet50;
inputSize = net.Layers(1).InputSize;
xlsxFile = 'C:\CS479_Final_Project\Images\Capture_Event_Groups\capture_event_groups.xlsx';
jsonFile = fileread('C:\CS479_Final_Project\Images\Bounding_Box_Output\Bounding_Boxes.json');
json = jsondecode(jsonFile);
unprocessedImagePath = 'C:\CS479_Final_Project\Images\Unprocessed_Sonoma_Cameratrap\';
processedImagePath = 'C:\CS479_Final_Project\Images\Processed_Sonoma_Cameratrap\';
outputName = "Cropped_Compensated_Normalized_5FoldPartition";
excludeList = {'crow'; 'hawk'; 'owl'; 'stellar_s jay'; 'unknown';};
camera = {'UpperROWWoodChipFieldCamera'; 'LowerTrailCamera'; 'NorthernTowerMeadowCamera'; 'SODPlotCamera'; 'UpperTrailCamera'; 'UpperMostROWCamera'};

% Image preprocessing for cropping images based on bounding boxes stored in
% a .json file produced by the microsoft animal detection neural net
% image_preprocessing(jsonFile, unprocessedImagePath, processedImagePath); % Only need to run this function once!

% Leave-One-Camera-Out Partition Code
% [imdsArray, valArray, groupSizes, cameraNames, numClasses] = createLOCOGroupings(processedImagePath, excludeList, camera);
% imdsHoldover = imdsArray{end,1};
% imdsArray(end) = [];
% [imdsArray, numClasses] = oversampling_compensation_for_advanced_partitions(imdsArray, false);
% imdsArray{end+1} = imdsHoldover;
% numfolds = numel(imdsArray);

% KFold Partition Code
numfolds = 5;
[imdsArray, valArray, groupSizes] = createKFoldGroupings(xlsxFile, processedImagePath, excludeList, numfolds);
%imdsHoldover = imdsArray{end,1};
%imdsArray(end) = [];
[imdsArray, numClasses] = oversampling_compensation_for_advanced_partitions(imdsArray, false);
%imdsArray{end+1} = imdsHoldover;

% Naive Partition Code
% imdsTrain = imageDatastore(processedImagePath,'IncludeSubfolders',true,'LabelSource','foldernames');
% [imdsTrain, imdsTest] = splitEachLabel(imdsTrain, 0.70);
% imdsArray{1,1} = imdsTrain;
% imdsArray = oversampling_compensation_for_advanced_partitions(imdsArray, false);
% imdsArray{2,1} = imdsTest;
% numClasses = numel(categories(imdsArray{1,1}.Labels));
% numfolds = 1;

[net, avg_fscore, avg_precision, avg_recall, avg_accuracy] = training_script(net, imdsArray, valArray, numClasses, inputSize, numfolds, outputName);

% net = active_learner();

%Grab embedding from penultimate layer
layer = 'fc1000_softmax';
embedFeatures = activations(net,augimdsTrain,layer,'OutputAs','rows');
