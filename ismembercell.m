function ism =ismembercell(mat, cell_arr)
% returns an array of size size(cell_array) with ism(i) = true if
% cell_arr{i} is 
    ism = cellfun(@(m)isequal(m,mat),cell_arr);
end