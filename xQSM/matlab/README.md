## For quick demo/evaluation of our network
Please directly run the '**run_demo.m**' script provided in '**eval**' folder. We have provided a human brain COSMOS and its local field map in this folder. Four pretrained networks (xQSM_invivo, U-net_invivo, xQSM_synthetic, U-net_synthetic) can be directly applied. **16G** of RAM recommended.

## To train a new xQSM network based on your data
Please add the '**train**' folder into your MATLAB path, and then run the scripts: '**TrainUnet.m**' and '**TrainXQSM.m**'. It is strongly recommended that the training should be performed on GPU, or on high performance computing clusters.
