% ZeroPadding to make the size of Field divisible by the designated factor;

function [Field, pos] = ZeroPadding(Field, factor)
    imSize = size(Field);
    upSize = ceil(imSize / factor) * factor; 
    pos_init = ceil((upSize - imSize) / 2) + 1;
    pos_end = pos_init + imSize - 1; 
    pos = [pos_init; pos_end];
    tmp_Field = zeros(upSize);
    tmp_Field(pos(1,1):pos(2,1), pos(1,2):pos(2,2), pos(1,3): pos(2,3)) = Field;
    Field = tmp_Field; 
end


