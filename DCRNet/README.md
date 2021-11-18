# DCRNet: Accelerating Quantitative Susceptibility Mapping using Compressed Sensing and Deep Neural Network

* This repository is for a deep complex residual network (DCRNet) to recover both MR magnitude and quantitative phase images from the CS undersample k-space data, enabling the acceleration of QSM acquisitions, which is introduced in the following paper: https://doi.org/10.1016/j.neuroimage.2021.118404 

- This code was built and tested on Centos 7.8 with Nvdia Tesla V100 and windows 10/ubuntu 19.10 with GTX 1060. 

* It is recommended that this code should be run on a linux desktop; however, On a windows OS, you can still reconstruct magnitude and phase images from the subsampled data, while the QSM reconstruction codes (from phase images) will not work. 

# Content
- [ Overview](#head1)
	- [(1) Overall Framework](#head2)
	- [(2) Data Flow in Networks](#head3)
- [ Manual](#head4)
	- [Requirements](#head5)
	- [Quick Start (using example data)](#head6)
	- [The Whole Reconstruction Pipeline (on your own data)](#head7)
	- [Train new DCRNet](#head8)

# <span id="head1"> Overview </span>

## <span id="head2">(1) Overall Framework </span>

![Whole Framework](https://www.dropbox.com/s/f729s5l2xvpwjfx/Figs_1.png?raw=1)
Fig. 1: Overview of the proposed QSM accelerating scheme.  

## <span id="head3">(2) Data Flow in Networks </span>

![Data Flow](https://www.dropbox.com/s/2519jlm4cr8g9cp/Figs_2.png?raw=1)
Fig. 2: The architecture of the proposed DCRNet, which is developed from a deep residual network backbone using complex convolutional operations.

# <span id="head4"> Manual </span>

## <span id="head5"> Requirements </span>

* For DL-based Magnitude and Phase Reconstruction  
    - Python 3.7 or later  
    - NVDIA GPU (CUDA 10.0)  
    - Anaconda Navigator (4.6.11) for Pytorch Installation
    - Pytorch 1.8 
    - MATLAB 2017b or later  
* For QSM PostProcessing from MRI Phase Data  
    - Hongfu Sun's QSM toolbox (https://github.com/sunhongfu/QSM)
    - FMRIB Software Library v6.0 (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)

## <span id="head6"> Quick Start (using example data) </span>
1. Clone this repository. 

```
    git clone https://github.com/YangGaoUQ/DCRNet.git
```
2. Install prerequisites (on linux system);
    1. Installl anaconda (https://docs.anaconda.com/anaconda/install/linux/); 
    2. open a terminal and create your conda environment to install Pytorch 1.8 and supporting packages (scipy); the following is an example
        ```
            conda create --name Pytorch
            conda activate Pytorch 
            conda install pytorch cudatoolkit=10.2 -c pytorch-nightly
            conda install scipy
        ```
3. Now you are ready for MRI magnitude and phase reconstructions, however, for QSM reconstruction, do not forget to install Hongfu Sun's QSM toolbox and FMRIB Software Library.   

4. Download the Example Data provided by the author (https://drive.google.com/file/d/1ycrafjCsxft69y58wZ5ZZ7C22eMVQ4LH/view?usp=sharing), then unzip and move the file ('kspace_example.mat') into folder './TestData/'. 

5. Open a new terminal, and run the following command, then you will find the corresponding MRI and QSM reconstructions in the folder 'MRI_QSM_recon'
```
    cd ~/DCRNet
    conda activate Pytorch
    matlab -nodisplay -r Demo_on_ExampleData
```

## <span id="head7"> The Whole Reconstruction Pipeline (on your own data) </span>
1. Preprocess your test data, using 'kSpaceSubsampling.m' provided in the folder './MatlabCodes/'. This function takes two variables as its input, i.e., the path to your k-space data file, and a designated file indentifier 'FileNo'; 
```matlab 
    matlab -nodisplay -r "kSpaceSubSampling(datapath, FileNo)"
```

2. Modify the Inference code (in folder './PythonCodes/')
    1. Open './PythonCodes/Inference.py' using your own IDE
    2. go to line 14, set variable 'File_No' to be your file identifier (e.g., 2, 3, 4 ...)
    4. save it as your own inference script file. 

3. Run the modified code directly using python 

```python
    python your_own_inference_script.py  
```

or using the matlab codes provided in folder './MatlabCodes/'

```matlab
    matlab -nodispaly -r "PythonRecon('../PythonCodes/your_own_inference_script.py')"
```

4. Run the postprocessing codes to obtain MRI magnitude and phase images from the network outputs; 
```matlab
    matlab -nodisplay -r "MRI_PhaseRecon(FileNo)"
``` 

5. Run the QSM reconstruction in folder './MatlabCodes/':
```matlab
    matlab -nodispaly -r QSM_Recon_From_Phase
```

## <span id="head8"> Train new DCRNet </span>
1. Prepare and preprocess your data with the code provided in folder 'MatlabCodes':
```matlab
    matlab -nodispaly -r PrepareTrainingData
```
2. Go to folder "../PythonCodes/" and run the folling code (this is a script for quick training, complicated full-version training scripts will be updated soon): 

```python 
    python TrainDCRNet.py
```

[⬆ top](#readme)
