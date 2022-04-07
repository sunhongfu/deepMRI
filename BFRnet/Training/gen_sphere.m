function one_sphere = gen_sphere(p_shape)
%GEN_ Summary of this function goes here
%   Detailed explanation goes here

sus_std = 0.5;  % standard deviation of susceptibility values
shape_size_min_factor = 0.05;
shape_size_max_factor = 0.25;

susceptibility_value = sus_std * randn();

shape_size_min = floor(p_shape(1) * shape_size_min_factor);
shape_size_max = floor(p_shape(1) * shape_size_max_factor);

random_size = randi([shape_size_min, shape_size_max]);

x_pos = randi([1, p_shape(1)]);
y_pos = randi([1, p_shape(2)]);
z_pos = randi([1, p_shape(3)]);

[xx, yy, zz] = meshgrid(1:p_shape(1), 1:p_shape(2), 1:p_shape(3));

shape_generated = ((xx - x_pos) .^ 2 + (yy - y_pos) .^ 2 + (zz - z_pos) .^ 2) < random_size .^ 2;

shape_generated = shape_generated * susceptibility_value; 

one_sphere = single(shape_generated);

end
