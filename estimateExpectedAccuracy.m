function [eacc, maxp] = estimateExpectedAccuracy(cpb, pzezo, zezoconfs)
if pzezo ==0
    [maxp] = max(cpb(:, :)); 
    eacc = mean(maxp);
else
    [maxp, argmaxp] = deal(nan(size(zezoconfs,1), size(cpb, 2)));
    for i=1:size(zezoconfs, 1)    
        [maxp(i, :), argmaxp(i, :)] = max(cpb(:, :, i));   
    end
    eacc = mean(sum(maxp.*pzezo));
end
end
