function [nprobs, edges] = calibration(probs, trueEv)
probs = reshape(probs, [], 1);
% % mbep = reshape(mbeProbs, [], 1);
% % mbp = reshape(mbProbs, [], 1);
trueEv = reshape(trueEv, [], 1);
% % 
[nPoints, edges, inds] = histcounts(probs, [0:0.2:1]);
% % [nPointsMBe, ~, indsMBe] = histcounts(mbep, [0:0.2:1]);
% % [nPointsMB, ~, indsMB] = histcounts(mbp, [0:0.2:1]);
% % 
edges = edges(2:end)-0.1;
for iEdge =1:length(edges)
    if nPoints(iEdge)==0
        continue;
    end
    nprobs(iEdge) = sum(trueEv(inds==iEdge))/nPoints(iEdge);
end
end
