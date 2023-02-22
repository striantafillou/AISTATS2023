function [cpha, cphabar]= overalp_cond_prob_hat(y, x, ze, zo, DeDs, DoDs, domainCounts)
% [cpha, cphabar]= overalp_cond_prob_hat(y, x, ze, zo, DeDs, DoDs, domainCounts)
% estimates p(y|do(x), ze, zo) from De and Do


if isempty(zo)
    cpha = cond_prob_mult_no_zeros(y, [x ze], [DeDs.data; DoDs.data], domainCounts);
else
    cpha = cond_prob_mult_no_zeros(y, [x ze zo], [DoDs.data], domainCounts);
end
    cphabar = cond_prob_mult_no_zeros(y, [x ze], [DeDs.data], domainCounts);
end
       