%Set up data and network
net = resnet50;
inputSize = net.Layers(1).InputSize;
xlsxFile = 'C:\CS479_Final_Project\Images\Capture_Event_Groups\capture_event_groups.xlsx';
jsonFile = fileread('C:\CS479_Final_Project\Images\Bounding_Box_Output\Bounding_Boxes.json');
json = jsondecode(jsonFile);
unprocessedImagePath = 'C:\CS479_Final_Project\Images\Unprocessed_Sonoma_Cameratrap\';
processedImagePath = 'C:\CS479_Final_Project\Images\Processed_Sonoma_Cameratrap\';
outputName = "Cropped_Compensated_5FoldPartition";

% Image preprocessing for cropping images based on bounding boxes stored in
% a .json file produced by the microsoft animal detection neural net
% image_preprocessing(jsonFile, unprocessedImagePath, processedImagePath); % Only need to run this function once!

% Leave-One-Camera-Out Partition Code
% [imdsArray, groupSizes, cameraNames, numClasses] = createLOCOGroupings(processedImagePath);

% KFold Partition Code
numfolds = 5;
[imdsArray, groupSizes] = createKFoldGroupings(xlsxFile, processedImagePath, numfolds);
[imdsArray, numClasses] = oversampling_compensation_for_advanced_partitions(imdsArray);

% Naive Partition Code
% imdsTrain = imageDatastore(processedImagePath,'IncludeSubfolders',true,'LabelSource','foldernames');
% imdsTrain = oversampling_compensation_for_naive_partitions(processedImagePath);
% [imdsTrain, imdsTest] = splitEachLabel(imdsTrain, 0.70);

%Find layers to replace
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
%numClasses = numel(categories(imdsTrain.Labels));

%Replace Layers
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:43) = freezeWeights(layers(1:43));
lgraph = createLgraphUsingConnections(layers,connections);
%analyzeNetwork(net)

%Process data fo input into CNN
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
%augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'DataAugmentation',imageAugmenter,"ColorPreprocessing","gray2rgb");
%augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsTest,"ColorPreprocessing","gray2rgb");

%Training Configuration
%miniBatchSize = 128; % Make this much smaller if you do not have a CUDA deivce
%valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
%delete(gcp('nocreate')); % CUDA Setting
%parpool('local', 2); % CUDA Setting
%options = trainingOptions('sgdm', ...
%    'ExecutionEnvironment', 'gpu', ... % CUDA Setting
%    'MiniBatchSize',miniBatchSize, ...
%    'MaxEpochs',20, ...
%    'InitialLearnRate',3e-2, ...
%    'Shuffle','every-epoch', ...
%    'ValidationData',augimdsValidation, ...
%    'ValidationFrequency',valFrequency, ...
%    'Verbose',false, ...
%    'LearnRateSchedule','piecewise', ...
%    'LearnRateDropPeriod', 5, ...
%    'LearnRateDropFactor', 0.1, ...
%    'Momentum', .92, ....
%    'Plots','training-progress');

%Train Network based on config
for i = 1 : numel(imdsArray)
    filesTrain = cell(1);
    for k = 1 : numel(imdsArray)
        if k ~= i
            filesTrain = vertcat(filesTrain, imdsArray{k,1}.Files);
        end
    end
    filesTrain(1) = [];
    imdsTrain = imageDatastore(filesTrain, 'LabelSource','foldernames');
    imdsTest = imdsArray{i,1};
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'DataAugmentation',imageAugmenter,"ColorPreprocessing","gray2rgb");
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsTest, "ColorPreprocessing","gray2rgb");
    miniBatchSize = 128; % Make this much smaller if you do not have a CUDA deivce
    valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
    %delete(gcp('nocreate')); % CUDA Setting
    %parpool('local', 2); % CUDA Setting
    options = trainingOptions('sgdm', ...
        'ExecutionEnvironment', 'gpu', ... % CUDA Setting
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',20, ...
        'InitialLearnRate',3e-2, ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',valFrequency, ...
        'Verbose',false, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod', 5, ...
        'LearnRateDropFactor', 0.1, ...
        'Momentum', .92, ....
        'Plots','training-progress');
    
    net = trainNetwork(augimdsTrain,lgraph,options);
    
    YTest = imdsTest.Labels;
    YTrain = imdsTrain.Labels;
    [YPred,probs] = classify(net, augimdsValidation);
    accuracy(i) = mean(YPred == YTest);
    
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
    avg_fscoreArray(i) = mean(fscore, 'omitnan')
    avg_PrecisionArray(i) = mean(precision, 'omitnan')
    avg_RacallArray(i) = mean(recall, 'omitnan')
    accuracy
end

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
mean(precision, 'omitnan')
mean(recall, 'omitnan')
accuracy

% Save network for later use
save(fullfile('C:\CS479_Final_Project\Output\Finished_Networks', append(outputName, '.network')), 'net');
saveas(fig, fullfile('C:\CS479_Final_Project\Output\Output_Charts', outputName), 'png');

%Grab embedding from penultimate layer
layer = 'fc1000_softmax';
embedFeatures = activations(net,augimdsTrain,layer,'OutputAs','rows');
