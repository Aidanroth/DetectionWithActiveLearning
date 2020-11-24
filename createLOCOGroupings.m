function [imdsArray, groupSizes, camera, numClasses] = createLOCOGroupings(filePath)
    camera = {};
    imds = imageDatastore(filePath, 'IncludeSubfolders', true);
    numClasses = numel(categories(imdsFull.Labels));
    for i = 1 : numel(imds.Files)
        [filepath, name, ext] = fileparts(imds.Files{i});
        newname = extractBefore(name, '_');
        camera{i} = newname;
    end
    camera = unique(camera);
    for i = 1 : numel(camera)
        grouping(:,i) = contains(imds.Files, camera(i));
        groupSizes(i) = sum(grouping(:,i));
        fileNames = imds.Files(grouping(:,i),1);
        imdsArray{i,1} = imageDatastore(fileNames,'LabelSource','foldernames');
    end
end
