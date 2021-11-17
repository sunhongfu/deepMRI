% inverse funciton of ZeroPadding. 

function [Field] = ZeroRemoving(Field, pos)
    Field = Field(pos(1,1):pos(2,1), pos(1,2):pos(2,2), pos(1,3): pos(2,3));
end

