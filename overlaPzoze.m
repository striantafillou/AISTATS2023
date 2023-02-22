function [condProb, yconfigs, zconfigs]= overlapPzoze(zo, ze, jtalg, domainCounts)
% [pygivz, configs]= estimateCondProbJT(y, z, jtalg, nVars, domainCounts)
% estimate probability y|z using junction tree (jtalg is a tetrad junction
% tree)
% configs: configurations of the conditioning set.
if isempty(zo)
    [condProb, yconfigs, zconfigs] = deal([], [], []);
end
if numel(zo)==1
    yconfigs = 0:domainCounts(zo)-1;
    if isempty(ze)
        condProb = jtalg.getMarginalProbability(zo-1);
        zconfigs=[];
        return;
    end

    nConfigs = prod(domainCounts(ze)); 
    condProb = nan(domainCounts(zo), nConfigs);

    pz = nan(1, nConfigs);
    zconfigs =  variableInstances(domainCounts(ze), false)-1;
    for iConfig =1:nConfigs
        pz(iConfig) = jtalg.getJointProbability(ze-1, zconfigs(iConfig,:));
        condProb(:, iConfig) = jtalg.getConditionalProbabilities(zo-1, ze-1, zconfigs(iConfig,:));
    end
else
    nyConfigs = prod(domainCounts(zo));
    yconfigs =  variableInstances(domainCounts(zo), false)-1;

    if isempty(ze)
        condProb = nan(prod(domainCounts(zo)), 1);
        for iyConfig =1:nyConfigs
            condProb(iyConfig) = jtalg.getJointProbability(zo-1, yconfigs(iyConfig, :));
        end
        zconfigs=[];
        return;
    end

    nConfigs = prod(domainCounts(ze)); 
    condProb = nan(prod(domainCounts(zo)), nConfigs);
    %pz = nan(1, nConfigs);
    zconfigs =  variableInstances(domainCounts(ze), false)-1;
    for iConfig =1:nConfigs
        %pz(iConfig) = jtalg.getJointProbability(z-1, zconfigs(iConfig,:));
        tmp = jtalg.getConditionalProbabilities(zo-1, ze-1, zconfigs(iConfig,:));
        condProb(:, iConfig) = tmp.getProbabilities;        
    end
    yconfigs = tmp.getValues;

end
% evidence = cell(1,nVars);
% if isempty(z)
%     [eng, ll] = enter_evidence(engine, evidence);
%      mnodes = marginal_nodes(eng, y);
%      pygivz = mnodes.T;
% else
%     nConfigs = prod(domainCounts(z));
%     pygivz=nan(domainCounts(y), nConfigs);
%     nZ = length(z);
%     configs =  variableInstances(domainCounts(z), false);
%     for iConfig =1:nConfigs
%         curConfig = configs(iConfig, :);
%         z_c =mat2cell(curConfig', ones(nZ,1));
%         evidence(z) = deal(z_c);
%         [eng, ll] = enter_evidence(engine, evidence);
%         mnodes = marginal_nodes(eng, y);
%         pygivz(:, iConfig) = mnodes.T;
%     end
% end