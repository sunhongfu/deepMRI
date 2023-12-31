%% configure the matlab and python linkage; 
pyExec = 'C:\Users\CSU\anaconda3\envs\Pytorch\python.exe'; % conda environment path (windows), replace it with yours;  
pyRoot = fileparts(pyExec);
p = getenv('PATH');
p = strsplit(p, ';');
addToPath = {
   pyRoot
   fullfile(pyRoot, 'Library', 'mingw-w64', 'bin')
   fullfile(pyRoot, 'Library', 'usr', 'bin')
   fullfile(pyRoot, 'Library', 'bin')
   fullfile(pyRoot, 'Scripts')
   fullfile(pyRoot, 'bin')
};
p = [addToPath(:); p(:)];
p = unique(p, 'stable');
p = strjoin(p, ';');
setenv('PATH', p);

% % clear classes
% module_to_load = 'numpy';
% python_module_to_use = py.importlib.import_module(module_to_load);
% py.importlib.reload(python_module_to_use);

% pe = pyenv('Version','D:\Users\ASUS\anaconda3\envs\torch13\python.exe'); 