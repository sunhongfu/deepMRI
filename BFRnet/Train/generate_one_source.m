function one_data = generate_one_source(p_shape)
% generate one single simulate data;
%
one_data = single(zeros(p_shape));
squares_total = randi([5, 10]); % number of squares;
rectangles_total = randi(5, 10);  % number of rectrangulars;
spheres_total = randi([5, 10]);    % number of spheres;
% % squares_total = randi([150, 350]); % number of squares;
% % rectangles_total = randi(150, 350);  % number of rectrangulars;
% % spheres_total = randi([150, 350]);    % number of spheres;
sus_std = 0.5;  % standard deviation of susceptibility values
shape_size_min_factor = 0.1;
shape_size_max_factor = 0.5;

%% simualte rectangulars;
for i = 1 : rectangles_total
    susceptibility_value = sus_std * randn();
    
    shape_size_min = floor(p_shape(1) * shape_size_min_factor);
    shape_size_max = floor(p_shape(1) * shape_size_max_factor);
    
    random_sizex = randi([shape_size_min, shape_size_max]);
    random_sizey = randi([shape_size_min, shape_size_max]);
    random_sizez = randi([shape_size_min, shape_size_max]);
    
    x_pos = randi([1, p_shape(1)]);
    y_pos = randi([1, p_shape(2)]);
    z_pos = randi([1, p_shape(3)]);
    
    x_pos_max = x_pos + random_sizex - 1;
    if x_pos_max >= p_shape(1)
        x_pos_max = p_shape(1);
    end
    
    y_pos_max = y_pos + random_sizey - 1;
    if y_pos_max >= p_shape(2)
        y_pos_max = p_shape(2);
    end
    
    z_pos_max = z_pos + random_sizez - 1;
    if z_pos_max >= p_shape(3)
        z_pos_max = p_shape(3);
    end
    
    rectangle_temp = zeros(p_shape);
    rectangle_temp(x_pos:x_pos_max, y_pos:y_pos_max, z_pos:z_pos_max) = susceptibility_value;
    
    angle = randi([0, 180]);
    axes = randi([0, 1]);
    rectangle_temp = imrotate3(rectangle_temp, angle, [0, axes, 1 - axes], 'crop');
    one_data = one_data + rectangle_temp;
end

%% simulate squares;
for i = 1 : squares_total
    susceptibility_value = sus_std * randn();
    
    shape_size_min = floor(p_shape(1) * shape_size_min_factor);
    shape_size_max = floor(p_shape(1) * shape_size_max_factor);
    
    random_size = randi([shape_size_min, shape_size_max]);

    x_pos = randi([1, p_shape(1)]);
    y_pos = randi([1, p_shape(2)]);
    z_pos = randi([1, p_shape(3)]);
    
    x_pos_max = x_pos + random_size - 1;
    if x_pos_max >= p_shape(1)
        x_pos_max = p_shape(1);
    end
    
    y_pos_max = y_pos + random_size - 1;
    if y_pos_max >= p_shape(2)
        y_pos_max = p_shape(2);
    end
    
    z_pos_max = z_pos + random_size - 1;
    if z_pos_max >= p_shape(3)
        z_pos_max = p_shape(3);
    end
    
    square_temp = zeros(p_shape);
    square_temp(x_pos:x_pos_max, y_pos:y_pos_max, z_pos:z_pos_max) = susceptibility_value;
    
    angle = randi([0, 180]);
    axes = randi([0, 1]);
    square_temp = imrotate3(square_temp, angle, [0, axes, 1 - axes], 'crop');
    one_data = one_data + square_temp;
end

%% simulate spheres;
for i = 1 : spheres_total
    sphere_temp = gen_sphere(p_shape);
    one_data = one_data + sphere_temp; 
end

one_data = single(one_data / 3); %% convert to float 32; 
end
