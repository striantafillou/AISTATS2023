function pydoxze = overlapAdjustment(pyxzezo, pzoze)
% takes as input p(y|x, ze, zo) in a y times x times zeconfigs times
% zoconfigs matrix, and p(zo|ze) in a zoconfigs times zeconfigs matrix and
% returns p(y|do(x), ze) = \sum_zo p(y|x, ze, zo)*p(zo|ze) in a y times x
% times zeconfigs matrix.
if isempty(pzoze)
    pydoxze = pyxzezo;
    return;
elseif size(pzoze,2)==1
    pydoxze = nan(size(pyxzezo,1), size(pyxzezo,2));
    for iX=1:size(pyxzezo, 2)
        pydoxze(:, iX) = sum(squeeze(pyxzezo(:,iX,:)).*pzoze(:)',2);
    end
else
    pydoxze = nan(size(pyxzezo,1), size(pyxzezo,2), size(pyxzezo, 3));
    for iZe =1:size(pyxzezo, 3)
        for iX=1:size(pyxzezo, 2)
            pydoxze(:, iX, iZe) = sum(squeeze(pyxzezo(:,iX, iZe, :)).*pzoze(:, iZe)',2);
        end
    end
end