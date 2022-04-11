function dataout = PreProcess(data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    imgs = data{:,1};
    labels = data{:,2};
    ll = size(data, 1);
    imgout = cell(ll, 1);
    labelout = cell(ll, 1);
    %% 
    for i = 1 : size(data, 1) 
        img = imgs{i, 1};
        label = labels{i, 1};
        msk = label ~= 0; 
        %% 
        imgout{i} = img .* msk; 
        labelout{i} = label .* msk;
    end
    dataout = table(imgout, labelout);
end

