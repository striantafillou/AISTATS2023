function [score,score_in] = overlapBDscore(counts, priors)
% dirichlet_bayesian_score calculates 
% log \int \prod_y P(D|theta_y) f(theta_i) d(theta_i)
% counts is the times Y=i [domainCounts(y)\times nConfigs vector]
% prior is an optional multidimensional array of the same shape as counts.
% It defaults to a uniform prior.
% Calculated as
% https://stephentu.github.io/writeups/dirichlet-conjugate-prior.pdf, page
% 6.

scores = nan(size(counts(2:end)));
if nargin==1
    priors = zeros(size(counts));
end
% for iX =1:size(counts,2)
%     for iZe =1:size(counts, 3)
%         n_j = sum(counts(:, iX, iZe));
%         n_jk = squeeze(counts(:, iX,iZe)); 
%         a_j = sum(priors(:, iX, iZe)+1);
%         a_jk = squeeze(priors(:, iX,iZe))+1;
%         scores(iX, iZe) = gammaln(a_j) - gammaln(n_j+a_j) + sum(gammaln(n_jk+a_jk))-sum(gammaln(a_jk));
%     end
% end
N = sum(counts);
score_in = gammaln(sum(priors+1))-gammaln(N+sum(priors+1))+sum(gammaln(counts+priors+1))-sum(gammaln(priors+1));
score = sum(sum(score_in));
%gammaln(dc)+sum(gammaln(counts+priors+1))- gammaln(sum(counts+priors+1));
end