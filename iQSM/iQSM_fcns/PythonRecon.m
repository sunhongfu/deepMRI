function PythonRecon(PyFile, InputPath, OutPath)
% descritions: 
% inputs: PyFile: Python script path; default: ../PythonCodes/Inference.py;
% 
if ~ exist('PyFile','var') || isempty(PyFile)
    PyFile = './PythonCodes/Evaluation/Inference.py';
end
%% Call Python script to conduct the reconstruction; 
curDir = pwd; 

if ispc
    ConfigPython;  %% configure pytorh for matlab terminal
end

disp('Calling Python for iQSM reconstruction'); 

%% cd ../PythonCodes/
[codeFolder, codeName, codeExt] = fileparts(PyFile); 
cd(codeFolder); 

CodeFile = [codeName codeExt]; 
exe_command = sprintf("python -u %s -I %s -O %s -C %s", CodeFile, InputPath, OutPath, CheckpointsPath);

system(exe_command); % !python -u Inference.py
cd(curDir)

disp('iQSM Reconstruction Finished');
end

