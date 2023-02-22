function [acc, y, yhat] = estimateAccuracy(cpb, zezoconfs, data)
yhat = nan(size(data, 1),1);
y = data(:, 2);

for i= 1:size(zezoconfs,1)
    %fprintf('--------------------%d-------------\n', i)
    patinds0 = ismember(data(:, [1 3:end]), [0 zezoconfs(i, :)], 'rows');
    [~, yhat(patinds0)]= max(cpb(:, 1, i));
   % mean(yhat(patinds0)-1 == y(patinds0))
    patinds1 = ismember(data(:, [1 3:end]), [1 zezoconfs(i, :)], 'rows');
    [~, yhat(patinds1)]= max(cpb(:, 2, i));
   %mean(yhat(patinds1)-1 == y(patinds1))
end
yhat = yhat-1;
acc = mean(y==yhat);
end