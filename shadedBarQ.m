function [ h ] = shadedBarQ( x, quants, color, opacity)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
x=reshape(x,1, length(x));

x1 = [x fliplr(x)];
%x(1)= x(1)+0.005; x(length(x))= x(length(x))-0.005;


y= [quants(:,1)' fliplr(quants(:,2)')];
h.fill = fill(x1, y, color, 'linestyle', 'none');
% Choose a number between 0 (invisible) and 1 (opaque) for facealpha.  

set(h.fill,'facealpha',opacity)
end

