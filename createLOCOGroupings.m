function [imdsArray, valArray, groupSizes, camera, numClasses] = createLOCOGroupings(filePath, excludeList, camera)
%     if empty(camera)
%         % Search for cameras
%         camera = {};
%         imds = imageDatastore(filePath, 'IncludeSubfolders', true);
%         numClasses = numel(categories(imds.Labels));
%         for i = 1 : numel(imds.Files)
%             [filepath, name, ext] = fileparts(imds.Files{i});
%             newname = extractBefore(name, '_');
%             camera{i} = newname;
%         end
%         camera = unique(camera);    
%     end
    imds = imageDatastore(filePath, 'IncludeSubfolders', true);
    numClasses = numel(unique(imds.Labels));
    for i = 1 : numel(camera)
        grouping(:,i) = contains(imds.Files, camera(i)) & ~contains(imds.Files, excludeList);
        groupSizes(i) = sum(grouping(:,i));
        fileNames = imds.Files(grouping(:,i),1);
        imdsArray{i,1} = imageDatastore(fileNames,'LabelSource','foldernames');
        valArray{i,1} = imageDatastore(fileNames,'LabelSource','foldernames');
    end
end
