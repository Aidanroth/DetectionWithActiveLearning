%Set up data and network
net = resnet50;
inputSize = net.Layers(1).InputSize;
xlsxFile = 'C:\CS479_Final_Project\Images\Capture_Event_Groups\capture_event_groups.xlsx';
jsonFile = fileread('C:\CS479_Final_Project\Images\Bounding_Box_Output\Bounding_Boxes.json');
json = jsondecode(jsonFile);
unprocessedImagePath = 'C:\CS479_Final_Project\Images\Unprocessed_Sonoma_Cameratrap\';
processedImagePath = 'C:\CS479_Final_Project\Images\Processed_Sonoma_Cameratrap\';
outputName = "Cropped_Compensated_5FoldPartition";
excludeList = {'crow'; 'hawk'; 'owl'; 'stellar_s jay'; 'unknown';};
camera = {'UpperROWWoodChipFieldCamera'; 'LowerTrailCamera'; 'NorthernTowerMeadowCamera'; 'SODPlotCamera'; 'UpperTrailCamera'; 'UpperMostROWCamera'};

% Image preprocessing for cropping images based on bounding boxes stored in
% a .json file produced by the microsoft animal detection neural net
% image_preprocessing(jsonFile, unprocessedImagePath, processedImagePath); % Only need to run this function once!

% Leave-One-Camera-Out Partition Code
% [imdsArray, groupSizes, cameraNames, numClasses] = createLOCOGroupings(processedImagePath, excludeList, camera);
% imdsHoldover = imdsArray{end,1};
% imdsArray(end) = [];
% [imdsArray, numClasses] = oversampling_compensation_for_advanced_partitions(imdsArray, false);
% imdsArray{end+1} = imdsHoldover;

% KFold Partition Code
numfolds = 5;
[imdsArray, groupSizes] = createKFoldGroupings(xlsxFile, processedImagePath, excludeList, numfolds);
imdsHoldover = imdsArray{end,1};
imdsArray(end) = [];
[imdsArray, numClasses] = oversampling_compensation_for_advanced_partitions(imdsArray, false);
imdsArray{end+1} = imdsHoldover;

% Naive Partition Code
% imdsTrain = imageDatastore(processedImagePath,'IncludeSubfolders',true,'LabelSource','foldernames');
% [imdsTrain, imdsTest] = splitEachLabel(imdsTrain, 0.70);
% imdsArray{1,1} = imdsTrain;
% imdsArray = oversampling_compensation_for_advanced_partitions(imdsArray(1));
% imdsArray{2,1} = imdsTest;
% numClasses = numel(categories(imdsArray{1,1}.Labels));

[net, augimdsTrain, augimdsValidation, imdsTest, imdsTrain] = training_script(net, imdsArray, numClasses, inputSize);

%Test network on testing set of images
YTest = imdsTest.Labels;
YTrain = imdsTrain.Labels;
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == YTest);

%Analyze results
c = confusionmat(YTest, YPred);
writematrix(c, fullfile('C:\CS479_Final_Project\Output\Confusion_Matrices', append(outputName, '.csv')));
fig = figure;
fig.Position(3) = fig.Position(3) * 2.5;
fig.Position(4) = fig.Position(4) * 1.5;
cm = confusionchart(YTest, YPred, "ColumnSummary",'column-normalized',"RowSummary","row-normalized","Title",outputName);
labels = unique(YTest);
numLabels = numel(labels);
fp = zeros(numLabels, 1);
tp = zeros(numLabels, 1);
fn = zeros(numLabels, 1);
tn = zeros(numLabels, 1);
for k = 1 : numLabels
    tp(k, 1) = sum((YTest == labels(k, 1)) & (YPred == labels(k, 1)));
    fp(k, 1) = sum((YTest ~= labels(k, 1)) & (YPred == labels(k, 1)));
    fn(k, 1) = sum((YTest == labels(k, 1)) & (YPred ~= labels(k, 1)));
    tn(k, 1) = sum((YTest ~= labels(k, 1)) & (YPred ~= labels(k, 1)));
end
precision = tp ./ (tp + fp);
recall = tp ./ (tp + fn);
fscore = (2 .* precision .* recall) ./ (precision + recall);
fscore = fscore';
avg_fscore = mean(fscore, 'omitnan')
avg_precision = mean(precision, 'omitnan')
avg_recall = mean(recall, 'omitnan')
accuracy

% Save network for later use
save(fullfile('C:\CS479_Final_Project\Output\Finished_Networks', append(outputName, '.network')), 'net');
saveas(fig, fullfile('C:\CS479_Final_Project\Output\Output_Charts', outputName), 'png');

%Grab embedding from penultimate layer
layer = 'fc1000_softmax';
embedFeatures = activations(net,augimdsTrain,layer,'OutputAs','rows');
