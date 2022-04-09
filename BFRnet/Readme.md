## For quick demo/evaluation of our network
Please directly run the '**BFRnet_demo.m**' script provided in '**eval**' folder. We have provided a human brain COSMOS and its tocal field map in this folder. The pre-trained BFRnet could be directly applied at /(Network address)/. **16G** of RAM recommended.

## Training dataset generation
Please add the '**train**' folder into your MATLAB path and run the scripts '**Gen_HighSus.m**'.
To generate the synthetic background susceptibility and field map, please view '**PhanGene.m**'

## BFRnet training based on your data
Please directly run the scripts: '**TrainOctNet130BFR3.m**'. It is strongly recommended that the training should be performed on GPU, or on high performance computing clusters.