function [condProb, zeconfigs]= overlapPyxz(y, x, ze, jtalg, domainCounts)
% [pygivz, configs]= overlapPyxz(y, x, ze, jtalg, domainCounts)
% estimate probability P(y|x, z) using junction tree (jtalg is a tetrad junction
% tree )
% configs: zeconfigurations of the adjustment set.
if isempty(ze)
    zeconfigs=[];
    condProb = nan(domainCounts(y), domainCounts(x));
    for iX =1:domainCounts(x)
        condProb(:, iX) = ...
            jtalg.getConditionalProbabilities(y-1, x-1, [iX-1]);
    end
else
    zeconfigs =  variableInstances(domainCounts(ze), false)-1;
    nzeConfigs  = size(zeconfigs,1);
    condProb = nan(domainCounts(y), domainCounts(x), nzeConfigs);
    for iX =1:domainCounts(x)
        for izeConfig =1:nzeConfigs
            condProb(:, iX,izeConfig) = ...
                jtalg.getConditionalProbabilities(y-1, [x ze]-1, [iX-1 zeconfigs(izeConfig,:)]);
        end
    end
end