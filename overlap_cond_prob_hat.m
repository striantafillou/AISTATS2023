function [cpha, cphabar,cpb, zezoconfigs, zeconfigs]= overlap_cond_prob_hat(y, x, ze, zo, DeDs, DoDs, domainCounts, probsha)
% [cpha, cphabar]= overalp_cond_prob_hat(y, x, ze, zo, DeDs, DoDs, domainCounts)
% estimates p(y|do(x), ze, zo) from De and Do

[phabar,~, xzeconfigs] = cond_prob_mult_inst(y, [x ze], [DeDs.data], domainCounts);
phabar = reshape(phabar, domainCounts(y), domainCounts(x), []);
nZe= length(ze);
if isempty(zo)
    [cpha, ~,xzezoconfigs] = cond_prob_mult_inst(y, [x ze], [DeDs.data; DoDs.data], domainCounts);
    cpha = reshape(cpha, domainCounts(y), domainCounts(x), []);
else
    [cpha, ~,xzezoconfigs] = cond_prob_mult_inst(y, [x ze zo], [DoDs.data], domainCounts);
    cpha = reshape(cpha, domainCounts(y), domainCounts(x), []);
end

zezoconfigs = unique(xzezoconfigs(:,2:end), 'rows', 'stable');
zeconfigs = unique(xzeconfigs(:,2:end), 'rows', 'stable');
nConfigs =size(zezoconfigs,1);
cphabar = nan(domainCounts(y), domainCounts(x), nConfigs);
for iConfig = 1:nConfigs
    izeconf = ismember(zeconfigs, zezoconfigs(iConfig, 1:nZe),'rows');
    cphabar(:, :, iConfig) = phabar(:, :, izeconf);
end
cpb =cpha*probsha(1)+cphabar*probsha(2);
end
       