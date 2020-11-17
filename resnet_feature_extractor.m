%Set up data and network
net = resnet50;
inputSize = net.Layers(1).InputSize;
jsonFile = fileread('C:\Users\aidan\OneDrive\Documents\CS 479\ObjDetectOutput\output1.json');
json = jsondecode(jsonFile);
baseImagePath = 'C:\Users\aidan\Documents\AnimalDetectionImages\images\test\';
outputImagePath = 'C:\Users\aidan\Documents\AnimalDetectionImages\images\testout\';
image_preprocessing_oversampled(jsonFile, baseImagePath, outputImagePath); % Only need to run this function once!
imdsTrain = imageDatastore(outputImagePath,'IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imdsTrain, 0.60);

%Find layers to replace
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
numClasses = numel(categories(imdsTrain.Labels));

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
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'DataAugmentation',imageAugmenter,"ColorPreprocessing","gray2rgb");
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsTest,"ColorPreprocessing","gray2rgb");

%Training Configuration
miniBatchSize = 64;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
% delete(gcp('nocreate'));
% parpool('local', 2);
% options = trainingOptions('sgdm', ...
%     'WorkerLoad', [.9 .9], ... %Change this setting for different CUDA device setups
%     'ExecutionEnvironment', 'multi-gpu', ...
%     'DispatchInBackground', true, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MaxEpochs',30, ...
%     'InitialLearnRate',3e-2, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',augimdsValidation, ...
%     'ValidationFrequency',valFrequency, ...
%     'Verbose',false, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod', 5, ...
%     'LearnRateDropFactor', 0.1, ...
%     'Momentum', .92, ...
%     'CheckpointPath', 'C:\CS479_Final_Project\Model_Checkpoints', ...
%     'Plots','training-progress');
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment', 'cpu', ... % CUDA Setting
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

%Train Network based on config
net = trainNetwork(augimdsTrain,lgraph,options);

%Test network on testing set of images
YTest = imdsTest.Labels;
YTrain = imdsTrain.Labels;
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == YTest)

%Analyze results
c = confusionmat(YTest, YPred);
writematrix(c, 'ResNetRetrainResults.csv');
fig = figure;
fig.Position(3) = fig.Position(3) * 2;
cm = confusionchart(YTest, YPred, "ColumnSummary",'column-normalized',"RowSummary","row-normalized","Title","Scene Classification with ResNet50");
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


%Grab embedding from penultimate layer
layer = 'fc1000_softmax';
embedFeatures = activations(net,augimdsTrain,layer,'OutputAs','rows');
