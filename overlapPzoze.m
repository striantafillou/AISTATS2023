function [condProb, zoconfigs, zeconfigs]= overlapPzoze(zo, ze, jtalg, domainCounts)
% [pygivz, configs]= overlapPzoze(y, z, jtalg, nVars, domainCounts)
% estimate probability zo|ze using junction tree (jtalg is a tetrad junction
% tree)
% configs: configurations of the conditioning set.
if isempty(zo)
    [condProb, zoconfigs, zeconfigs] = deal([], [], []);
    return;
end

if numel(zo)==1
    zoconfigs = 0:domainCounts(zo)-1;
    if isempty(ze)
        condProb = jtalg.getMarginalProbability(zo-1);
        zeconfigs=[];
        return;
    end

    nConfigs = prod(domainCounts(ze)); 
    condProb = nan(domainCounts(zo), nConfigs);

    pz = nan(1, nConfigs);
    zeconfigs =  variableInstances(domainCounts(ze), false)-1;
    for iConfig =1:nConfigs
        pz(iConfig) = jtalg.getJointProbability(ze-1, zeconfigs(iConfig,:));
        condProb(:, iConfig) = jtalg.getConditionalProbabilities(zo-1, ze-1, zeconfigs(iConfig,:));
    end
else
    nyConfigs = prod(domainCounts(zo));
    zoconfigs =  variableInstances(domainCounts(zo), false)-1;

    if isempty(ze)
        condProb = nan(prod(domainCounts(zo)), 1);
        for iyConfig =1:nyConfigs
            condProb(iyConfig) = jtalg.getJointProbability(zo-1, zoconfigs(iyConfig, :));
        end
        zeconfigs=[];
        return;
    end

    nConfigs = prod(domainCounts(ze)); 
    condProb = nan(prod(domainCounts(zo)), nConfigs);
    %pz = nan(1, nConfigs);
    zeconfigs =  variableInstances(domainCounts(ze), false)-1;
    for iConfig =1:nConfigs
        %pz(iConfig) = jtalg.getJointProbability(z-1, zconfigs(iConfig,:));
        tmp = jtalg.getConditionalProbabilities(zo-1, ze-1, zeconfigs(iConfig,:));
        condProb(:, iConfig) = tmp.getProbabilities;        
    end
    zoconfigs = tmp.getValues;

end
