function [trueAsP, possets] =computePriors(dag, isLatent, nVars)

    mag = dag2mag(dag, isLatent);
    mag = mag(1:nVars, 1:nVars);
    oracleds.data = mag;
    oracleds.isAncestor = findancestors(mag);
    oracleds.type = 'oracle';
    oracleds.isLatent= isLatent(1:nVars);
    pag = FCI(oracleds, 'verbose', 0, 'maxK',5, 'alpha', 0.01, 'cons', true);


    % estimate P(Hz|Do)
    [possets, isPosAS, keepVars] = isPossibleAdjustmentSet12(pag, 1);
    
    nSets = size(possets, 1);
    noBidir= false;
    if isPosAS(nSets+1)==false
        noBidir=true;
    end
    trueAsP = false(1, nSets);
    for iSet =1:nSets
        curSet = find(possets(iSet, :));
        trueAsP(iSet) = isAdjustmentSet(1, 2, curSet, dag);
    end
    trueAsP(nSets+1) = ~any(trueAsP(1:nSets));
    nisAdjSet = false;
    if trueAsP(nSets+1)
        nisAdjSet=true;
    end
    times(iter, 2) = toc(t);
end
