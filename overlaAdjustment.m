function pydoxze = overlaAdjustment(pyxzezo, pzoze)
% takes as input p(y|x, ze, zo) in a y times x times zeconfigs times
% zoconfigs matrix, and p(zo|ze) in a zoconfigs times zeconfigs matrix and
% returns p(y|do(x), ze) = \sum_zo p(y|x, ze, zo)*p(zo|ze) in a y times x
% times zeconfigs matrix.
pydoxze = nan(size(pyxzezo,1), size(pyxzezo,2), size(pzoze, 3));

for iZe =1:size(pzoze, 3)
    for iX=1:size(pzoze, 2)
        pydoxze(:, iX, iZe) = sum(squeeze(pyxzezo(:,iX, iZe, :)).*pzoze(:, iZe)',2);
    end
end