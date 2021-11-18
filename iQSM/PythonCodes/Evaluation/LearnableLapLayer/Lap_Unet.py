
#################### Section 1 ###########################
## Import necessary library and packages and further pre-
## -defined packages here.
## import self-defined packages
from Unet import * 
################# End Section 1 ##########################

#################### Section 2 ###########################
## Parameters： Encoding depth: Times of Poolings 
class Lap_Unet(nn.Module):
    def __init__(self, Lap_Layer, Unet_part):
        super(Lap_Unet, self).__init__()
        self.Unet = Unet_part
        self.Lap_Layer = Lap_Layer

    def forward(self, wphs, masks, TEs, B0):
        ## input： x, wrapped phase images; 
        
        Lap_Filtered_results, LearnableFilterd_results = self.Lap_Layer(wphs, masks, TEs, B0)

        recon = self.Unet(Lap_Filtered_results, LearnableFilterd_results)

        recon = recon / 4 ## simple linear normalization due to training settings; 

        return recon
