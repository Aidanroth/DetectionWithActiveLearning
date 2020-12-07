%Set up data and network
net = resnet50;
inputSize = net.Layers(1).InputSize;
xlsxFile = 'C:\CS479_Final_Project\Images\Capture_Event_Groups\capture_event_groups.xlsx';
jsonFile = fileread('C:\CS479_Final_Project\Images\Bounding_Box_Output\Bounding_Boxes.json');
json = jsondecode(jsonFile);
unprocessedImagePath = 'C:\CS479_Final_Project\Images\Unprocessed_Sonoma_Cameratrap\';
processedImagePath = 'C:\CS479_Final_Project\Images\Processed_Sonoma_Cameratrap\';
outputName = "Cropped_Compensated_Normalized_5FoldPartition_Test2";
excludeList = {'crow'; 'hawk'; 'owl'; 'stellar_s jay'; 'unknown';};
camera = {'UpperROWWoodChipFieldCamera'; 'LowerTrailCamera'; 'NorthernTowerMeadowCamera'; 'SODPlotCamera'; 'UpperTrailCamera'; 'UpperMostROWCamera'};

% Image preprocessing for cropping images based on bounding boxes stored in
% a .json file produced by the microsoft animal detection neural net
% image_preprocessing(jsonFile, unprocessedImagePath, processedImagePath); % Only need to run this function once!

% Leave-One-Camera-Out Partition Code
% [imdsArray, valArray, groupSizes, cameraNames, numClasses] = createLOCOGroupings(processedImagePath, excludeList, camera);
% [imdsArray, numClasses] = oversampling_compensation_for_advanced_partitions(imdsArray, false);
% numfolds = numel(imdsArray);

% KFold Partition Code
numfolds = 5;
[imdsArray, valArray, groupSizes] = createKFoldGroupings(xlsxFile, processedImagePath, excludeList, numfolds);
[imdsArray, numClasses] = oversampling_compensation_for_advanced_partitions(imdsArray, false);

% Naive Partition Code
% imdsTrain = imageDatastore(processedImagePath,'IncludeSubfolders',true,'LabelSource','foldernames');
% [imdsTrain, imdsTest] = splitEachLabel(imdsTrain, 0.70);
% imdsArray{1,1} = imdsTrain;
% imdsArray = oversampling_compensation_for_advanced_partitions(imdsArray, false);
% imdsArray{2,1} = imdsTest;
% numClasses = numel(categories(imdsArray{1,1}.Labels));
% numfolds = 1;

[net, avg_fscore, avg_precision, avg_recall, avg_accuracy, embedFeatures] = training_script(net, imdsArray, valArray, numClasses, inputSize, numfolds, outputName);


% net = load('C:\CS479_Final_Project\Output\Finished_Networks\Cropped_Compensated_Normalized_5FoldPartition_F5.MAT');
% filesTrain = cell(1);
% for k = 1 : numel(imdsArray)
%     if k ~= 5
%         filesTrain = vertcat(filesTrain, imdsArray{k,1}.Files);
%     end
% end
% filesTrain(1) = [];
% imdsTrain = imageDatastore(filesTrain, 'LabelSource','foldernames');
% imdsTest = valArray{5,1};
% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsTest, "ColorPreprocessing","gray2rgb");
% YTest = imdsTest.Labels;
% YTrain = imdsTrain.Labels;
% [YPred, probs] = classify(net.net,augimdsValidation);
% accuracy = mean(YPred == YTest);
% 
% idx = randperm(numel(imdsTest.Files),16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(imdsTest,idx(i));
%     imshow(I)
%     label = YPred(idx(i));
%     title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
% end
% net = active_learner();