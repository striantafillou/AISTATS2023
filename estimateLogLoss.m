function [logloss, lls] = estimateLogLoss(cpb, zezoconfs, data)

lls = nan(size(data, 1),1);
y = data(:, 2);

for i= 1:size(zezoconfs,1)
    patinds0 = ismember(data(:, [1 3:end]), [0 zezoconfs(i, :)], 'rows');
    y0s = y(patinds0);py = cpb(2, 1,i)+2*eps; 
    lls(patinds0) = -y0s.*log(py)-(1-y0s).*log(1-py);

    patinds1 = ismember(data(:, [1 3:end]), [1 zezoconfs(i, :)], 'rows');
    y1s = y(patinds1);py = cpb(2, 2,i)+2*eps;
    lls(patinds1) = -y1s.*log(py)-(1-y1s).*log(1-py);
end
logloss =nansum(lls)./length(y);
end