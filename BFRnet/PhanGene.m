%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate one geometric-bkg with the size same as the input image;
% 'matrix' refers to the matrix size brain image (e.g., 144*192*128 voxwls)
% 'R' refers to the radius of geometric kernels like clinders, shpere and cube.
% Default radius is 20 voxel.
% We set 200 geometric kernels in each shape. 600 in totall. 
% The output 'bkg' refers to the entire background susceptibility kernels.
% 'Coor...' refers to the center coordinate of each kernel. 
% Susceptibility refers to the susceptibility value of each kernel.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bkg,CoorVec,CoorVecS,CoorVecCL,Susceptibility] = PhanGene(matrix,R,Number)
% [lfsSphere, mask_ero_Sphere] = resharp(field_sphere,mask_sphere);
%% Global Prameters

sizeA = size(matrix); size1 = sizeA(1); size2 = sizeA(2); size3 = sizeA(3); 
A = zeros(size1,size2,size3);
size_edge1 = size1 - R*2; size_edge2 = size2 - R*2; size_edge3 = size3 - R*2; 
% Number = 200; the number of geometric shapes;
CoorVec = zeros(Number,4);
CoorVecS = zeros(Number,4);
CoorVecCL = zeros(Number,4);
Susceptibility = zeros(Number,3);
% Try to set an ultra matrix to store these 256^3Cubes in 4-D matrix %
%%
for num = 1:1:Number
   %% Variable Prameters 
    RandCoord1 = round(rand(1,3)*size1); % Set x coordinate
    RandCoord2 = round(rand(1,3)*size2); % Set y coordinate
    RandCorrd3 = round(rand(1,3)*size3); % Set z coordinate
    
    CoorVec(num,1) = num; % Loop number
    CoorVec(num,2:4) = [RandCoord1(1),RandCoord2(1),RandCorrd3(1)]; % x1,y1,z1
    Coor_X = CoorVec(num,2); Coor_Y = CoorVec(num,3); Coor_Z = CoorVec(num,4); % Cube Initial Coordinate
    SusVal = (rand(1,3)*10-2);  % rand(0,1) refers to uniform distribution between 0~1
    Susceptibility(num,1) = SusVal(1);%  Suceptibility Value, Sphere
    
    % RandCoordS = round(rand(1,3)*size1);
    CoorVecS(num,1) = num; % Loop number
    CoorVecS(num,2:4) = [RandCoord1(2),RandCoord2(2),RandCorrd3(2)]; % x2,y2,z2
    CoorS_X = CoorVecS(num,2); CoorS_Y = CoorVecS(num,3); CoorS_Z = CoorVecS(num,4); % Sphere Initial Coordinate
    Susceptibility(num,2) = SusVal(2);% Random Setting Uniform Suceptibility Value, Cube
    
    % RandCoordCL = round(rand(1,3)*size1);
    CoorVecCL(num,1) = num; % Loop number
    CoorVecCL(num,2:4) = [RandCoord1(3),RandCoord2(3),RandCorrd3(3)]; % x3,y3,z3
    CoorCL_X = CoorVecCL(num,2); CoorCL_Y = CoorVecCL(num,3); CoorCL_Z = CoorVecCL(num,4); % Cliner Initial Coordinate
    Susceptibility(num,3) = SusVal(3); % Random Setting Uniform Suceptibility Value, Clinder

    %% Sphere Loop
    for k = 1:1:size3
        for j = 1:1:size2
            for i = 1:1:size1
                if sqrt((i-CoorS_X)^2+(j-CoorS_Y)^2+(k-CoorS_Z)^2) <= R % Sphere
                   A(i,j,k) = SusVal(1);
                end
            end
        end
    end
    %% Clinder Loop
    for l = 1:1:size1
        for m = 1:1:size2
            if sqrt((l-CoorCL_X)^2+(m-CoorCL_Y)^2) <= R && (CoorCL_Z<=size_edge3)% Clinder
               A(l,m,CoorCL_Z+1:CoorCL_Z+R*2) = SusVal(3);
            else
                if sqrt((l-CoorCL_X)^2+(m-CoorCL_Y)^2) <= R && (CoorCL_Z>size_edge3)% Clinder
                   A(l,m,CoorCL_Z+1:size3) = SusVal(3);
                end
            end
        end
    end
  %% Cube Switch Cases ( 0:<= / 1:> )
  if   Coor_X<=size_edge1 && Coor_Y<=size_edge2 && Coor_Z<=size_edge3 % 000
     A((Coor_X+1:Coor_X+R*2),(Coor_Y+1:Coor_Y+R*2),(Coor_Z+1:Coor_Z+R*2)) = SusVal(2);
    else
  if Coor_X<=size_edge1 && Coor_Y<=size_edge2 && Coor_Z>size_edge3    % 001
      A((Coor_X+1:Coor_X+R*2),(Coor_Y+1:Coor_Y+R*2),(Coor_Z+1:size3)) = SusVal(2);
    else
  if Coor_X<=size_edge1 && Coor_Y>size_edge2 && Coor_Z<=size_edge3    % 010
      A((Coor_X+1:Coor_X+R*2),(Coor_Y+1:size2),(Coor_Z+1:Coor_Z+R*2)) = SusVal(2);
    else
  if Coor_X<=size_edge1 && Coor_Y>size_edge2 && Coor_Z>size_edge3    % 011
      A((Coor_X+1:Coor_X+R*2),(Coor_Y+1:size2),(Coor_Z+1:size3)) = SusVal(2);
    else
  if Coor_X>size_edge1 && Coor_Y<=size_edge2 && Coor_Z<=size_edge3    % 100
      A((Coor_X+1:size1),(Coor_Y+1:Coor_Y+R*2),(Coor_Z+1:Coor_Z+R*2)) = SusVal(2);
    else
  if Coor_X>size_edge1 && Coor_Y<=size_edge2 && Coor_Z>size_edge3     % 101
      A((Coor_X+1:size1),(Coor_Y+1:Coor_Y+R*2),(Coor_Z+1:size3)) = SusVal(2);
    else
  if Coor_X>size_edge1 && Coor_Y>size_edge2 && Coor_Z<=size_edge3     % 110
      A((Coor_X+1:size1),(Coor_Y+1:size2),(Coor_Z+1:Coor_Z+R*2)) = SusVal(2);
    else
  if Coor_X>size_edge1 && Coor_Y>size_edge2 && Coor_Z>size_edge3     % 111
      A((Coor_X+1:size1),(Coor_Y+1:size2),(Coor_Z+1:size3)) = SusVal(2);
  end
  end
  end
  end
  end
  end
  end
  end
end
bkg = A;
end
