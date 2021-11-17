function OpenFolder(dir)
%OPENFOLDER Summary of this function goes here

if isunix
    system(['xdg-open', ' ', dir]);
elseif ismac
    system(['open',' ', dir]); 
elseif ispc
    winopen(dir)
end

end

