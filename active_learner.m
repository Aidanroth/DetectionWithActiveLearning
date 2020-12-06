function output = active_learner()

    %unzip('rabbit_squirrel.zip');
    d = gpuDevice
    %imds = imageDatastore('D:/Downloads/rabbit_squirrel/rabbit_squirrel', ...
        %'IncludeSubfolders',true, ...
        %'LabelSource','foldernames');

    %[imdsTrain,imdsValidation] = splitEachLabel(imds,0.5,'randomized');

    %%%Initialize_Network%%%
    %net = alexnet;
    %inputSize = net.Layers(1).InputSize;
    %layersTransfer = net.Layers(1:end-3);
    %numClasses = numel(categories(imdsTrain.Labels));
    %classes = {'rabbit', 'squirrel'};
    %layers = [
        %layersTransfer
        %fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        %softmaxLayer
        %classificationLayer('Classes',classes)];


    %%%Train_Network%%%
    %pixelRange = [-30 30];
    %imageAugmenter = imageDataAugmenter( ...
        %'RandXReflection',true, ...
        %'RandXTranslation',pixelRange, ...
        %'RandYTranslation',pixelRange);
    %augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        %'DataAugmentation',imageAugmenter);
    %augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
    %options = trainingOptions('sgdm', ...
        %'MiniBatchSize',10, ...
        %'MaxEpochs',1, ...
        % 'InitialLearnRate',1e-4, ...
        % 'Shuffle','every-epoch', ...
        % 'ValidationData',augimdsValidation, ...
        % 'ValidationFrequency',20, ...
        % 'Verbose',false, ...
        % 'Plots','training-progress');
    %netTransfer = trainNetwork(augimdsTrain,layers,options);




    for i = 1:10
    
        %%%Classify_Unlabeled_Images%%%
        sz = 100;
        imdsPredict = imageDatastore('D:/rabbit_squirrel_extra', 'LabelSource', 'foldernames');
        pixelRange = [-30 30];
        imageAugmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXTranslation',pixelRange, ...
            'RandYTranslation',pixelRange);
        augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsPredict, ...
            'DataAugmentation',imageAugmenter);
        [YPred,scores] = classify(netTransfer, augimdsTrain);
        imdsPredict.Labels = imdsPredict.Files;

   
   


        %%%Sort_Images_By_Condifence%%%
        score_filename = strings(sz(1),2);
        for j = 1:sz(1)
            score_filename(j,1) = scores(j,1);
            score_filename(j,2) = imdsPredict.Labels(1);
        end
        score_array = str2double(score_filename(:,1));
        [sortedValues, sortOrder] = sort(score_array);

    
    
    

        %%%Manually_Label_Low_Confidence_Results_And_Send_To_Other_Folder%%%
        %%%sorted by confidence is low confidence first, therfore use first
        %%%half for manual labels. Second half are high confidence images.
        labels = strings(sz/2);
        for j = 1 : sz/2
            filename = score_filename(sortOrder(j),2); 
            image = imread(filename);
            imshow(image);
            labels(j,1) = input('Label: ');
        end
        for j = 1 : sz/2
            %movefile(filename, 'D:/rabbit_squirrel_extra_used'); change
        end
        for j = sz/2 : 1
            filename = score_filename(sortOrder(j),2);
            %movefile(filename, 'D:/rabbit_squirrel_extra_used') ;change
        end

    
    
    

        %%%Train_Network_On_New_Data%%%
        %%%Network should train on folder classified images were sent to. First
        %%% create datastore, classify
        imdsGoodTrain = imageDatastore('D:/rabbit_squirrel_extra_used', 'LabelSource', 'foldernames');
        [YPred,scores] = classify(netTransfer, imdsGoodTrain);
        imdsGoodTrain.Labels = YPred;
        for j = sz/2:1
            imdsGoodTrain.Labels(sortOrder(j)) = cellstr(labels(j));
        end
        pixelRange = [-30 30];
        imageAugmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXTranslation',pixelRange, ...
            'RandYTranslation',pixelRange);
        augimdsGoodTrain = augmentedImageDatastore(inputSize(1:2),imdsGoodTrain, ...
            'DataAugmentation',imageAugmenter);
        imdsGoodtrain.Labels = labels(:,1);
        netTransfer = trainNetwork(augimdsGoodTrain,layers,options);
    
    
    
    
    
        %%%Delete Old Files%%%
        filePattern = fullfile('D:/rabbit_squirrel_extra_used/', '*.jpg'); % Change to whatever pattern you need.
        theFiles = dir(filePattern);
        for k = 1 : length(theFiles)
            baseFileName = theFiles(k).name;
            fullFileName = fullfile('D:/rabbit_squirrel_extra_used/', baseFileName);
            delete(fullFileName);
        end
    end
end







