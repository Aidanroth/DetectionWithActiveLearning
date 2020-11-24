function imdsFull = oversampling_compensation_for_naive_partitions(croppedImagePath)
% This function has the implementation for oversampling compensation
% assuming a naive partitioning is desired. The function takes as input a
% path to the directory of images and returns a compensated imageDatastore.

    subfoldersToExclude = {'nothing'}; % Specify subfolder names that you do not wish to be involved in the oversampling compensation
    subfolderNames = dir(croppedImagePath);
    subfolderNames = subfolderNames([subfolderNames.isdir]);
    subfolderNames = subfolderNames(~ismember({subfolderNames.name}, {'.','..', subfoldersToExclude{1 : end}}));
    pathsToInclude = cell(numel(subfolderNames),1);
    pathsToExclude = cell(numel(subfoldersToExclude),1);
    for i = 1 : numel(subfolderNames)
        pathsToInclude{i} = fullfile(croppedImagePath, subfolderNames(i).name, '\');
    end
    for i = 1 : numel(subfoldersToExclude)
        pathsToExclude{i} = fullfile(croppedImagePath, subfoldersToExclude{i}, '\');
    end
    rPaths = matlab.io.datastore.DsFileSet(pathsToInclude);
    ePaths = matlab.io.datastore.DsFileSet(pathsToExclude);
    
    % Create two datastores, one to process, one to be left alone
    imdsIncluded = imageDatastore(rPaths, 'IncludeSubfolders',true,'LabelSource','foldernames');
    imdsExcluded = imageDatastore(ePaths, 'IncludeSubfolders',true,'LabelSource','foldernames');
    Labels = imdsIncluded.Labels;
    
    % Begin processing the datastore
    % Might want to try included the excluded label counts when determining
    % the desired number of observations for the set that is meant to be
    % modified. This allows for folders to be excluded from the
    % modification process, but still have their counts impact the number
    % of observations. Basically, you get a greater degree of granularity
    % from this implementation.
    [G, classes] = findgroups(Labels);
    numObservations = splitapply(@numel,Labels,G);
    numObservations = sort(numObservations, 'descend');
    desiredNumObservationsPerClass = numObservations(2); % Change this value to determine how many files you want replicated. Will grab the nth largest group size
    % desiredNumObservationsPerClass = round(mean(numObservations)); % This might be a more viable option for determining the normalization quota
    files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},imdsIncluded.Files,G);
    files = vertcat(files{:});
    Labels=[];
    info=strfind(files,'\');
    
    for i=1:numel(files)
        idx=info{i};
        dirName=files{i};
        targetStr=dirName(idx(end-1)+1:idx(end)-1);
        targetStr2=cellstr(targetStr);
        Labels=[Labels;categorical(targetStr2)];
    end
    
    imdsIncluded.Files = files;
    imdsIncluded.Labels = Labels;
    
    % Concatenate the processed and unprocessed datastore into a single
    % datastore to return to the main funciton
    imdsFull = imageDatastore({});
    imdsFull.Files = vertcat(imdsIncluded.Files, imdsExcluded.Files);
    imdsFull.Labels = vertcat(imdsIncluded.Labels, imdsExcluded.Labels);
end

function files = randReplicateFiles(files,numDesired)
    n = numel(files);
    ind = randi(n,numDesired,1);
    files = files(ind);
end