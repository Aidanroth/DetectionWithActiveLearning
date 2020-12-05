function [imdsArray, groupSizes] = createKFoldGroupings(tableFile, filePath, excludeList, n)
    A = readtable(tableFile);
    for i = 1 : n
        grouping(:,i) = (rem(A.Var2, n) == i - 1) & (~contains(A.Var1, excludeList)); % Grouping based on table
        fileNames = A(grouping(:,i),1); % Extract filename
        groupSizes(i) = numel(fileNames); % Get num images in group
        for k = 1 : numel(fileNames) % For each image in group
            groupingFiles{k,i} = fileNames{k,1}{:,:}; % Pull file name up one level
            groupingFiles{k,i} = append(filePath, groupingFiles{k,i}); % create absolute path
            [path, name, ext] = fileparts(groupingFiles{k,i}); % separate absolute path into subparts
            groupingFiles{k,i} = append(path, '\', name, '*', ext); % Add wildcard chars to grab cropped images with modified namescheme

        end
    end
    for i = 1 : n
        imdsArray{i,1} = imageDatastore(groupingFiles(1:groupSizes(i),i),'LabelSource','foldernames'); % Create datastores based on groups generated above
        groupSizes(i) = numel(imdsArray{i,1}.Files); % update size of group based on new datastore
    end
end

