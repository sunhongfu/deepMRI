function data = matRead(filename)
% data = matRead(filename) reads the image data in the MAT-file filename
% data = matRead(filename) reads the image data in the NII-file filename

nii = load_untouch_nii(filename);
data = nii.img;

end