function [imdsArray, numClasses] = oversampling_compensation_for_advanced_partitions(imdsArray, excludeFlag)
% This function will support oversampling compensation for a K-Fold
% partition and for a Leave-One-Camera-Out partition by taking as input an
% array of imageDatastores and compensating for those based on the average
% number of images in each file. The function should not reduce the number
% of files within any given folder, only increase the small folders to the
% size of the average.

    filesFull = imdsArray{1,1}.Files;
    for i = 2 : numel(imdsArray)
        filesFull = vertcat(filesFull, imdsArray{i,1}.Files);
    end
    imdsFull = imageDatastore(filesFull, 'LabelSource','foldernames');
    Labels = imdsFull.Labels;
    numClasses = numel(categories(imdsFull.Labels));
    
    % Begin processing the datastore
    [G, classes] = findgroups(Labels);
    numObservations = splitapply(@numel,Labels,G);
    numObservations = sort(numObservations, 'descend');
    desiredNumObservationsPerClass = round(mean(numObservations)) / numel(imdsArray);
    % desiredNumObservationsPerClass = numObservations(3) / numel(imdsArray);
    
    % Do these steps for each imds in the imdsArray
    for k = 1 : numel(imdsArray)
        if(excludeFlag == true)
            table = countEachLabel(imdsArray{k,1});
            excludeIdx = table.Count > desiredNumObservationsPerClass;
            excludeLabels = table.Label(excludeIdx(:),1);
            SkippedLabels = imdsArray{k,1}.Labels(ismember(imdsArray{k,1}.Labels, excludeLabels, 'rows'));
            SkippedFiles = imdsArray{k,1}.Files(ismember(imdsArray{k,1}.Labels, excludeLabels, 'rows')); % Skipped Files
            Labels = imdsArray{k,1}.Labels(~ismember(imdsArray{k,1}.Labels, excludeLabels, 'rows'));
            files = imdsArray{k,1}.Files(~ismember(imdsArray{k,1}.Labels, excludeLabels, 'rows')); % Used Files
        else
            Labels = imdsArray{k,1}.Labels;
            files = imdsArray{k,1}.Files;
        end
        [G, classes] = findgroups(Labels);
        files = splitapply(@(x){randReplicateFiles(x, desiredNumObservationsPerClass)}, files, G);
        files = vertcat(files{:});
        Labels=[];
        info=strfind(files,'\');
    
        for i = 1 : numel(files)
            idx=info{i};
            dirName=files{i};
            targetStr=dirName(idx(end-1)+1:idx(end)-1);
            targetStr2=cellstr(targetStr);
            Labels=[Labels;categorical(targetStr2)];
        end
        if(excludeFlag == true)
            Labels = vertcat(Labels, SkippedLabels);
            files = vertcat(files, SkippedFiles);
        end
        imdsArray{k,1}.Files = files;
        imdsArray{k,1}.Labels = Labels;
    end
    
end

function files = randReplicateFiles(files,numDesired)
    n = numel(files);
    ind = randi(n,numDesired,1);
    files = files(ind);
end