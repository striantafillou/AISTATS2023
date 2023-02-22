function nocpt = noisyorcpt(xs, vs, leak)
% function nocpt = noisyorcpt(x)
% Leaky noisy or cpt : P(y = 1|x1, ..., xn) = 1- leak*\Prod 
% assumes all variables are binary
n = length(xs);
if nargin ==1
    vs = rand(1, n);
    leak = rand(1);
end
Bn = ind2subv(2*ones(1,n), 1:(2^n))-1;  % all n bit vectors, with the left most column toggling fastest (LSB)
cpt = zeros(2^n, 2);
q = (1-vs).^Bn;

cpt(:,2) = 1-prod(q,2).*leak;
cpt(:,1) = 1-cpt(:,2);
nocpt = reshape(cpt', numel(cpt), 1);
nocpt = reshape(nocpt, 2*ones(1,n+1));
end