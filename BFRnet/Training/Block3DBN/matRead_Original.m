function data = matRead(filename)
% data = matRead(filename) reads the image data in the MAT-file filename

inp = load(filename);
f = fields(inp);
data = inp.(f{1});
end