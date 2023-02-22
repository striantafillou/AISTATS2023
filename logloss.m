function ll = logloss(ps, ys)
    ll= -ys.*log(ps)-(1-ys).*log(1-ps);
end