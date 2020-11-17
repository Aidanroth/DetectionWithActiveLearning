function image_preprocessing_oversampled(jsonFile, baseImagePath, outputImagePath)
% This function will crop all the images in the data set according to the
% Bounding Boxes stored in the json file produced by the Microsoft Animal
% Detection Neural Network. The hope is that this will replicate the file
% structure of the original datastore.

    imdsFull = imageDatastore(baseImagePath,'IncludeSubfolders',true,'LabelSource','foldernames');
    Labels = imdsFull.Labels;
    
    labelCount = countEachLabel(imdsFull);histogram(imdsFull.Labels);title('label frequency')
    labels=imdsFull.Labels;
    [G,classes] = findgroups(labels);
    numObservations = splitapply(@numel,labels,G);
    desiredNumObservationsPerClass = max(numObservations);
    files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},imdsFull.Files,G);
    files = vertcat(files{:});
    labels=[];info=strfind(files,'\');
    for i=1:numel(files)
        idx=info{i};
        dirName=files{i};
        targetStr=dirName(idx(end-1)+1:idx(end)-1);
        targetStr2=cellstr(targetStr);
        labels=[labels;categorical(targetStr2)];
    end
    imdsFull.Files = files;
    imdsFull.Labels=labels;
    labelCount_oversampled = countEachLabel(imdsFull)
    
    json = jsondecode(jsonFile);
    % Extract json data
    numImagesAfterCrops = 1;
    numImages = numel(json.images);
    jsonFilenames = cell(numImages, 1);
    for i = 1 : numImages
        jsonFilenames{i} = json.images(i).file;
        if numel(json.images(i).detections) == 0 % Case for no detections
            jsonBbox(i,1) = 0;
            numImagesAfterCrops = numImagesAfterCrops + 1;
        else
            for j = 1 : numel(json.images(i).detections) % Num bounding boxes
                numImagesAfterCrops = numImagesAfterCrops + 1;  
                for k = 1 : numel(json.images(i).detections(j).bbox) % num coordinates
                    jsonBbox(i,j,k) = json.images(i).detections(j).bbox(k); 
                end
            end
        end
    end
    
    for i = 1 : numel(Labels) % Check if directories exist
        if ~exist(append(outputImagePath, string(Labels(i))), 'dir')
            mkdir(append(outputImagePath, string(Labels(i))));
        end
    end
    
    for i = 1 : numel(Labels) % Check if directories exist
        if ~exist(append(outputImagePath, string(Labels(i))), 'dir')
            mkdir(append(outputImagePath, string(Labels(i))));
        end
    end
    
    % formula for cropping:
    % I = current image, might be able to be referenced by
    %                                           jsonFilenames(curImageIdx, :);
    % x = jsonBbox(curImageIdx, curBBoxIdx, 1)*numel(I(1,:,1))
    % y = jsonBbox(curImageIdx, curBBoxIdx, 2)*numel(I(:,1,1))
    % width = jsonBbox(curImageIdx, curBBoxIdx, 3)*numel(I(1,:,1))
    % height = jsonBbox(curImageIdx, curBBoxIdx, 4)*numel(I(:,1,1))
    % croppedImage = imcrop(I, [ x y width height ])
    
    for i = 1 : numImages % For each image
        curImage = read(imdsFull); % read image from datastore
        [curPath, curFileName, curFileType] = fileparts(jsonFilenames{i});
        resX = numel(curImage(1,:,1)); % get x-resolution
        resY = numel(curImage(:,1,1)); % get y-resolution
        if (numel(json.images(i).detections) >= 1) % and (Labels(i) ~= 'nothing') % Not sure if we want this second condition
            for j = 1 : numel(json.images(i).detections) % for each detection in current image
                cropImage = imcrop(curImage, [ jsonBbox(i,j,1)*resX jsonBbox(i,j,2)*resY jsonBbox(i,j,3)*resX jsonBbox(i,j,4)*resY ] ); % crop according to bbox
                imwrite(cropImage, fullfile(outputImagePath, string(Labels(i)), append(curFileName, '_', string(j), curFileType))); % write new image to dir
            end
        else
            imwrite(curImage, fullfile(outputImagePath, string(Labels(i)), append(curFileName, '_1', curFileType))); % write old image to dir as there is no animal in image
        end
    end
end

function files = randReplicateFiles(files,numDesired)
n = numel(files);
ind = randi(n,numDesired,1);
files = files(ind);
end