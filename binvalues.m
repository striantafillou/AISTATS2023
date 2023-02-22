function [mean_bin, std_bin, bin_size, bin_centers, q_bin] = binvalues(values, binner, numBins)

[bin_size,bin_centers, whichBin] = histcounts(binner, numBins);
bin_width =[bin_centers(2:end)-bin_centers(1:end-1)]./2;
bin_centers = bin_centers(1:end-1)+bin_width;
nBins = max(whichBin);

for iBin =1:nBins
    mean_bin(iBin) = median(values(whichBin==iBin));
    std_bin(iBin) = std(values(whichBin==iBin));  
    q_bin(iBin,:) = quantile(values(whichBin==iBin), [.1, .90]);
end
end