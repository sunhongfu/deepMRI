function PythonRecon(PyFile)
% descritions: 
% inputs: PyFile: Python script path; default: ../PythonCodes/Inference.py;
% 
if ~ exist('PyFile','var') || isempty(PyFile)
    PyFile = '../PythonCodes/Inference.py';
end
%% Call Python script to conduct the reconstruction; 
curDir = pwd; 

if ispc
    ConfigPython; 
end

disp('Calling Python for DCRNet-based MRI reconstruction'); 

%% cd ../PythonCodes/
[codeFolder, codeName, codeExt] = fileparts(PyFile); 
cd(codeFolder); 

CodeFile = [codeName codeExt]; 
exe_command = ['python -u ', CodeFile];

system(exe_command); % !python -u Inference.py
cd(curDir)

disp('DCRNet-based Reconstruction Finished');
end

