function PythonRecon(PyFile, InputPath, OutPath, CheckpointsPath)
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

% getAbsFolderPath = @(y) string(unique(arrayfun(@(x) x.folder, dir(y), 'UniformOutput', false)));

% InputPath = getAbsFolderPath(InputPath); 

disp('Calling Python for QSM reconstruction'); 

%% cd ../PythonCodes/
% [codeFolder, codeName, codeExt] = fileparts(PyFile); 
% cd(codeFolder); 
% 
% CodeFile = [codeName codeExt]; 
% exe_command = sprintf('python -u %s -I %s -O %s -C %s', CodeFile, InputPath, OutPath, CheckpointsPath);
exe_command = sprintf('python -u %s -I %s -O %s -C %s', PyFile, InputPath, OutPath, CheckpointsPath);

system(exe_command); % !python -u Inference.py
cd(curDir)

end


