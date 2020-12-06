function [avg_fscore, avg_precision, avg_recall, avg_accuracy] = calc_results(net, imdsTrain, imdsTest, augimdsValidation, i, outputName)
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
    avg_fscore = mean(fscore, 'omitnan');
    avg_precision = mean(precision, 'omitnan');
    avg_recall = mean(recall, 'omitnan');
    avg_accuracy = accuracy;

    % Save network for later use

    save(fullfile('C:\CS479_Final_Project\Output\Finished_Networks', append(outputName, '_F', int2str(i), '.network')), 'net');
    saveas(fig, fullfile('C:\CS479_Final_Project\Output\Output_Charts', outputName), 'png');
    
end