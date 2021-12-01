%% configure the matlab and python linkage; 
pyExec = 'D:\Users\92381\Anaconda3\envs\Pytorch\'; % conda environment path (windows), replace it with yours;  
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