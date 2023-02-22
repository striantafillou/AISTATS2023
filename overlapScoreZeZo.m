function [logscoreha, logscoresha, thetas, logscores] = overlapScoreZeZo(y, x, ze, zo,jtalgs, domainCounts, nyxze, nyxze_do)
% nyxze_do:  from observational
% nyxze: true in De
I = size(jtalgs,1);
[nY, nX] = deal(domainCounts(y), domainCounts(x));
logscoresha = nan(I, 1);
% if isempty(ze)&&isempty(zo)
%     logscoreha =sum(dirichlet_bayesian_score(nyxze, nyxze_do));
%     [logscoresha, thetas, logscores] = deal(nan);
%     return;
% end
if isempty (ze)
    thetas = nan(domainCounts(y), domainCounts(x), I);
    logscores = nan(nX, I);
    for iter =1:I
        pyxzezo = overlapPyxzezo(y, x, ze, zo, jtalgs(iter), domainCounts);
        pzoze = overlapPzoze(zo, ze, jtalgs(iter), domainCounts);
        pydoxze =  overlapAdjustment(pyxzezo, pzoze); 
        
        thetas(:, :, iter) = pydoxze;
        logscores(:, iter) = squeeze(sum(nyxze.*log(pydoxze)));
        logscoresha(iter) = sum(sum(logscores(:,  iter)));
    end
elseif isempty(zo)

    logscoreha =sum(dirichlet_bayesian_score(nyxze, nyxze_do));
    [logscoresha, thetas, logscores] = deal(nan);
    return;
else
    thetas = nan(nY, nX, prod(domainCounts(ze)), I);
    logscores = nan(nX, prod(domainCounts(ze)), I);
    for iter =1:I
        pyxzezo = overlapPyxzezo(y, x, ze, zo, jtalgs(iter), domainCounts);
        pzoze = overlapPzoze(zo, ze, jtalgs(iter), domainCounts);
        pydoxze =  overlapAdjustment(pyxzezo, pzoze);
        thetas(:, :, :, iter) = pydoxze;
        logscores(:, :, iter) = squeeze(sum(nyxze.*log(pydoxze)));
        logscoresha(iter) = sum(sum(logscores(:, :, iter)));
    end
end
logscoreha = sumOfLogsV(logscoresha)-log(I);
%logscoreshbara =  sum(dirichlet_bayesian_score(nyxze));

end

