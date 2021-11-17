# Instant tissue field and magnetic susceptibility mapping from MRI raw phase using Laplacian enabled deep neural networks

* This repository is for a large-stencil Laplacian preprocessed deep learning-based neural network for near instant quantitative field and susceptibility mapping (i.e., iQFM and iQSM), enabling a single-step (end-to-end) local field and QSM reconstrcution from the raw MRI phase images, which is introduced in this paper: https://arxiv.org/abs/2111.07665 (under review). 

- This code was built and tested on Centos 7.8 with Nvdia Tesla V100 and macos12.0.1/ubuntu 19.10 with GTX 1060. 

# Content
- [ Overview](#head1)
	- [(1) Overall Framework](#head2)
	- [(2) Representative Results](#head3)
- [ Manual](#head4) 
	- [Requirements](#head5)
    - [Codes Description](#head9)
	- [Quick Start (using example data)](#head6)
	- [Reconstruction on your own data](#head7)
	- [Train new iQSM and iQFM networks](#head8)

# <span id="head1"> Overview </span>

## <span id="head2">(1) Overall Framework </span>

![Whole Framework](https://github.com/sunhongfu/deepMRI/blob/master/iQSM/Figs/Figs_1.png)
Fig. 1: Overview of iQFM and iQSM framework using the proposed Lap-Unet architecture, composed of a tailored Lap-Layer and a 3D residual Unet.

## <span id="head3">(2) Representative Results </span>

![Representative Results](https://github.com/sunhongfu/deepMRI/blob/master/iQSM/Figs/Figs_2.png)
Fig. 2: Comparison of different QSM methods on three ICH patients. Susceptibility images of two orthogonal views are illustrated for each subject. Red arrows point to the artifacts near the hemorrhage sources in different QSM reconstructions. 

# <span id="head4"> Manual </span>

## <span id="head5"> Requirements </span>

    - Python 3.7 or later  
    - NVDIA GPU (CUDA 10.0)  
    - Anaconda Navigator (4.6.11) for Pytorch Installation
    - Pytorch 1.8 or later
    - MATLAB 2017b or later 
    - Hongfu Sun's QSM toolbox (https://github.com/sunhongfu/QSM)
    - BET tool from FSL tool box 

## <span id="head9"> Codes Description </span>
    
    Two main demo codes showing how to use iQSM for QSM reconstruction: 
        - Demo_on_ExampleData.m ---- Complete reconstruction pipeline on a COSMOS based single-echo simulated data. 
        - Demo_on_Example_MEData.m ---- Complete reconstruction pipeline for a in vivo multi-echo MRI phase data. 

    matlab files: 
        * Simulating datasets for network training or evaluation:
            - 3D_Laplacian_Operator.mat ---- the 27-stencil point Laplacian kernel
            - Simulate_Wrapped_Phase_From_QSM.m ---- simulation pipeline 
            - cropQSMs.m ---- generate QSM and backgound patches
            - GenerateHealthyPatches.m ---- generate healthy training patches
            - Gen_HemoCal.m ---- generate pathological training patches from healthy patches
            - generate_one_source.m ---- generate one synthetic data from basic geometric shapes
        
        * network reconstruction: 
            - save_Input.m ---- save phase, B0, TE, and mask information as a single .mat file for network reconstruction
            - PythonRecon ---- Calling Pytorch file "Inference.py" for reconstruction

    pytorch codes: 
        - Inference.py ---- Pytorch API for iQSM and iQFM reconstruction
        - Unet.py and Unet_blocks.py ---- implementation of Lap-Unet
        - TrainingDataLoad.py ---- data loader during training 
        - TrainiQSM.py  ---- network training: iQSM
        - TrainiQFM.py  ---- network training: iQFM
        - TrainiQFM_and_iQSM.py  ---- train iQSM and iQFM simultaneously with data fidelity loss 
        - TrainiQFM_and_iQSM_16c.py  ---- network training (more learnable kernels in Lap-Layer)

## <span id="head6"> Quick Start (using example data) </span>
1. Clone this repository. 

```
    git clone https://github.com/YangGaoUQ/iQSM.git
```
2. Install prerequisites (on linux system);
    1. Installl anaconda (https://docs.anaconda.com/anaconda/install/linux/); 
    2. open a terminal and create your conda environment to install Pytorch and supporting packages (scipy); the following is an example
        ```
            conda create --name Pytorch
            conda activate Pytorch 
            conda install pytorch cudatoolkit=10.2 -c pytorch
            conda install scipy
        ```
3. Download our demo data from google drive https://drive.google.com/file/d/1-UtEDQ_8gtUC1WFgVInE3IaEZx4e9pjk/view?usp=sharing 

3. Open a new terminal, and run the following command, then you will find QSM reconstructions in the folder './QSM_recons'
```
    cd ./iQSM
    conda activate Pytorch
    matlab -nodisplay -r Demo_on_SinlgeEcho_DemoData
    matlab -nodisplay -r Demo_on_MultiEcho_DemoData
```

## <span id="head7"> Reconstruction on your own data </span>

1. For single-echo data: 
    Replace the parameters in the "Demo_on_SinlgeEcho_DemoData.m" (line 10-20) with yours. 
    then run the matlab code
    ```
        cd ./iQSM
        conda activate Pytorch
        matlab -nodisplay -r Demo_on_SinlgeEcho_DemoData
    ```
2. For multi-echo data:
    Replace the parameters in the "Demo_on_MultiEcho_DemoData.m" (line 10-20) with yours. 
    then run the matlab code
    ```
        cd ./iQSM
        conda activate Pytorch
        matlab -nodisplay -r Demo_on_MultiEcho_DemoData
    ```

## <span id="head8"> Train new networks </span>
1. Prepare and preprocess your data with the code provided in folder 'iQSM_fcns':
```matlab
    matlab -nodispaly -r PrepareFullSizedImages.m 
    matlab -nodispaly -r cropQSMs.m 
    matlab -nodispaly -r GenerateHealthyPatches.m
    matlab -nodispaly -r Gen_HemoCal.m
```
2. Go to folder 'PythonCodes' and run the training codes: 

```python 
    python TrainiQSM.py or ...
    python TrainiQFM.py or ...
    python TrainiQFM_and_iQSM.py or ...
    python TrainiQFM_and_iQSM_16c.py 
```

[⬆ top](#readme)
