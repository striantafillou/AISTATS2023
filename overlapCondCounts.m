function nyxze= overlapCondCounts(y, x, ze, zeconfigs, dataset, domainCounts)
% FUNCTION  nyxze = overlpaCondCounts(y, x, ze, dataset)
% Counts of Y given X=x, Ze=ze in data set De 
% nyxze : nY times nX times nzeConfigs matrix

if nargin==5
    data = dataset.data;
    domainCounts = dataset.domainCounts;
else
    data=dataset;
end
if isempty(ze)
    nyxze = nan(domainCounts(y), domainCounts(x));
    for iX =1:domainCounts(x)
        curData = data(ismember(data(:, x),iX-1), :);
        nyxze(:, iX) = histc(curData(:, y), [1:domainCounts(y)]-1);%+1;
    end
else
    nzeConfigs = size(zeconfigs, 1);
    nyxze = nan(domainCounts(y), domainCounts(x), nzeConfigs);
    for iX =1:domainCounts(x)
        for iZe =1:nzeConfigs
            curData = data(ismember(data(:, [x ze]),[iX-1, zeconfigs(iZe, :)], 'rows'), :);
            nyxze(:, iX, iZe) = histc(curData(:, y), [1:domainCounts(y)]-1);%+1;
        end
    end
end

end


