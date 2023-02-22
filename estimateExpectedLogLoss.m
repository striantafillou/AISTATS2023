function [logloss, lls] = estimateExpectedLogLoss(cpb, pzezo, zezoconfs)
if pzezo ==0
     lls = sum(-cpb.*log(cpb+eps));
     logloss = 0.5*sum(lls);
else
    lls = deal(nan(size(zezoconfs,1), size(cpb, 2)));
    for i=1:size(zezoconfs, 1)    
        lls(i, :) = sum(-cpb(:, :, i).*log(cpb(:, :, i)+eps));   
    end
    logloss = 0.5*sum(sum(lls,2).*pzezo);
end
end
