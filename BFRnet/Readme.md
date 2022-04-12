## For quick demo/evaluation of BFRnet
Please directly run the '**BFRnet_demo.m**' script provided in '**eval**' folder. We have provided a human brain COSMOS and its tocal field map in this folder. **16G** of RAM recommended. The pre-trained BFRnet and demo data are available at 
https://www.dropbox.com/sh/q678oapc65evrfa/AADh2CGeUzhHh6q9t3Fe3fVVa?dl=0

## Training dataset generation
Please add the '**train**' folder into your MATLAB path and run the scripts '**Gen_HighSus.m**'.
To generate the synthetic background susceptibility and field map, please view '**PhanGene.m**'

## BFRnet training based on your data
Please directly run the scripts: '**TrainBFRnet.m**'. It is strongly recommended that the training should be performed on GPU, or on high performance computing clusters.
