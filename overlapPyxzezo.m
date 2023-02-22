function [condProb, zeconfigs, zoconfigs]= overlapPyxzezo(y, x, ze, zo, jtalg, domainCounts)
% [pygivz, configs]= estimateCondProbJT(y, z, jtalg, nVars, domainCounts)
% estimate probability y|z using junction tree (jtalg is a tetrad junction
% tree)
% configs: configurations of the conditioning set.
if isempty(zo)
    [condProb, zeconfigs]= overlapPyxz(y, x, ze, jtalg, domainCounts);
    zoconfigs =[];
    return;
end
if isempty(ze)
    zeconfigs =[];
    nzoConfigs = prod(domainCounts(zo)); 
    condProb = nan(domainCounts(y), domainCounts(x),  nzoConfigs);

    zoconfigs =  variableInstances(domainCounts(zo), false)-1;

    for iX =1:domainCounts(x)
        for izoConfig =1:nzoConfigs
            condProb(:, iX,izoConfig) = ...
                jtalg.getConditionalProbabilities(y-1, [x zo]-1, [iX-1  zoconfigs(izoConfig,:)]);
        end
    end
else
    nzoConfigs = prod(domainCounts(zo)); 
    nzeConfigs = prod(domainCounts(ze));

    condProb = nan(domainCounts(y), domainCounts(x), nzeConfigs, nzoConfigs);

    zeconfigs =  variableInstances(domainCounts(ze), false)-1;
    zoconfigs =  variableInstances(domainCounts(zo), false)-1;

    for iX =1:domainCounts(x)
        for izeConfig =1:nzeConfigs
            for izoConfig =1:nzoConfigs
                condProb(:, iX,izeConfig,  izoConfig) = ...
                    jtalg.getConditionalProbabilities(y-1, [x ze zo]-1, [iX-1 zeconfigs(izeConfig,:) zoconfigs(izoConfig,:)]);
            end
        end
    end
end