function [net, avg_fscore, avg_precision, avg_recall, avg_accuracy, embedFeatures] = training_script(net, imdsArray, valArray, numClasses, inputSize, numfolds, outputName)

    %Find layers to replace
    if isa(net,'SeriesNetwork') 
        lgraph = layerGraph(net.Layers); 
    else
        lgraph = layerGraph(net);
    end 
    [learnableLayer,classLayer] = findLayersToReplace(lgraph);


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

    %Process data fo input into CNN
    pixelRange = [-30 30];
    scaleRange = [0.9 1.1];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange, ...
        'RandXScale',scaleRange, ...
        'RandYScale',scaleRange);
    
    for i = 1 : numfolds

        filesTrain = cell(1);
        if numel(imdsArray) > 2
            for k = 1 : numel(imdsArray)
                if k ~= i
                    filesTrain = vertcat(filesTrain, imdsArray{k,1}.Files);
                end
            end
        else
            filesTrain = imdsArray{1,1}.Files;
        end
        filesTrain(1) = [];
        imdsTrain = imageDatastore(filesTrain, 'LabelSource','foldernames');
        imdsTest = valArray{i,1};
        augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'DataAugmentation',imageAugmenter,"ColorPreprocessing","gray2rgb");
        augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsTest, "ColorPreprocessing","gray2rgb");
        miniBatchSize = 192; % Make this much smaller if you do not have a CUDA deivce
        valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
        %delete(gcp('nocreate')); % CUDA Setting
        %parpool('local', 2); % CUDA Setting
        options = trainingOptions('sgdm', ...
            'ExecutionEnvironment', 'gpu', ... % CUDA Setting
            'MiniBatchSize',miniBatchSize, ...
            'MaxEpochs', 10, ...
            'InitialLearnRate',1e-2, ...
            'Shuffle','every-epoch', ...
            'ValidationData',augimdsValidation, ...
            'ValidationFrequency',valFrequency, ...
            'Verbose',false, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod', 3, ...
            'LearnRateDropFactor', 0.1, ...
            'Momentum', .92, ....
            'Plots','training-progress');
    
        net = trainNetwork(augimdsTrain,lgraph,options);
        [avg_fscore(i), avg_precision(i), avg_recall(i), avg_accuracy(i), embedFeatures] = calc_results(net, imdsTrain, imdsTest, augimdsValidation, i, outputName);
    
    
    end
    avg_fscore = mean(avg_fscore);
    avg_precision = mean(avg_precision);
    avg_recall = mean(avg_recall);
    avg_accuracy = mean(avg_accuracy);

end