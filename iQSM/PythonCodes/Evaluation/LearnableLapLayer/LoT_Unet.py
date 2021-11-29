
#################### Section 1 ###########################
## Import necessary library and packages and further pre-
## -defined packages here.
## import self-defined packages
from Unet import * 
################# End Section 1 ##########################

#################### Section 2 ###########################
## Parameters： Encoding depth: Times of Poolings 
class LoT_Unet(nn.Module):
    def __init__(self, LoT_Layer, Unet_part):
        super(LoT_Unet, self).__init__()
        self.Unet = Unet_part
        self.LoT_Layer = LoT_Layer

    def forward(self, wphs, masks, TEs, B0):
        ## input： x, wrapped phase images; 
        
        LoT_Filtered_results, LearnableFilterd_results = self.LoT_Layer(wphs, masks, TEs, B0)

        recon = self.Unet(LoT_Filtered_results, LearnableFilterd_results)

        recon = recon / 4 ## simple linear normalization due to training settings; 

        return recon
