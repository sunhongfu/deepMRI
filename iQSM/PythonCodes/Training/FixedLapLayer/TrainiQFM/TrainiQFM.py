################### train AutoBCS framework #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from Unet import * 
from TrainingDataLoad import * 
#########  Section 1: DataSet Load #############
def DataLoad(Batch_size):
    DATA_DIRECTORY = '..'
    DATA_LIST_PATH = './test_IDs.txt'
    
    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def SaveNet(Unet_chi, epo, enSave = False):
    print('save results')
    #### save the
    if enSave:
        pass
    else:
        torch.save(Unet_lfs.state_dict(), './iQFM_UnetPart_final.pth')
        torch.save(Unet_lfs.state_dict(), ("./iQFM_UnetPart_%s.pth" % epo))

def TrainNet(Unet_chi, LR = 0.001, Batchsize = 32, Epoches = 100 , useGPU = True):
    print('IniReconNet')
    print('DataLoad')
    trainloader = DataLoad(Batchsize)
    print('Dataload Ends')

    print('Training Begins')
    criterion = nn.MSELoss(reduction='sum')

    optimizer1 = optim.Adam(Unet_chi.parameters())
    #ptimizer2 = optim.Adam(Unet_lfs.parameters())
    ##optimizer3 = optim.Adam(LPLayer.parameters())

    LGOP =  scio.loadmat("3D_Laplacian_Operator.mat", verify_compressed_data_integrity=False)
    conv_op = LGOP['LM']
    conv_op = np.array(conv_op)
    conv_op = torch.from_numpy(conv_op)
    conv_op = conv_op.float()
    conv_op = torch.unsqueeze(conv_op, 0)
    conv_op = torch.unsqueeze(conv_op, 0)

    LPLayer = LapLayer(conv_op)


    scheduler1 = LS.MultiStepLR(optimizer1, milestones = [50, 80], gamma = 0.1)
    #scheduler2 = LS.MultiStepLR(optimizer2, milestones = [50, 80], gamma = 0.1)
    ##scheduler3 = LS.MultiStepLR(optimizer3, milestones = [50, 80], gamma = 0.1)
    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            Unet_chi = nn.DataParallel(Unet_chi)
            Unet_chi.to(device)

            LPLayer = nn.DataParallel(LPLayer)
            LPLayer.to(device)

            for epoch in range(1, Epoches + 1):

                if epoch % 20 == 0:
                    SaveNet(Unet_chi, epoch, enSave = False)

                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    wphs, chis, TEs, masks, name = data
                    wphs = wphs.to(device)
                    chis = chis.to(device)
                    masks = masks.to(device)
                    TEs = TEs.to(device)
                    ## zero the gradient buffers 
                    optimizer1.zero_grad()
                    #optimizer2.zero_grad()
                    #optimizer3.zero_grad()
                    ## forward: 
                    b_i = LPLayer(wphs, masks, TEs)

                    pred_chi = Unet_chi(b_i)

                    pred_chi = pred_chi / 4

                    #pred_lfs = Unet_lfs(wphs)
                    ## loss
                    loss1 = criterion(pred_chi * masks, chis * masks)
                    ## backward
                    #loss2.backward(retain_graph = True)
                    loss1.backward()
                    ##
                    optimizer1.step()
                    #optimizer2.step()
                    ##optimizer3.step()

                    optimizer1.zero_grad()
                    #optimizer2.zero_grad()
                    ##optimizer3.zero_grad()
                    ## print statistical information 
                    ## print every 20 mini-batch size
                    if i % 19 == 0:
                        acc_loss1 = loss1.item()   
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss1: %f, lr1: %f,  used time: %d s' %
                            (epoch, i + 1, acc_loss1, optimizer1.param_groups[0]['lr'], time_end - time_start))  
                scheduler1.step()
                #scheduler2.step()
                ##scheduler3.step()
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')
    SaveNet(Unet_chi, Epoches)

if __name__ == '__main__':
    ## load laplacian operator; 
    """
    LGOP =  scio.loadmat("3D_Laplacian_Operator.mat", verify_compressed_data_integrity=False)
    conv_op = LGOP['LM']
    conv_op = np.array(conv_op)
    conv_op = torch.from_numpy(conv_op)
    conv_op = conv_op.float()
    conv_op = torch.unsqueeze(conv_op, 0)
    conv_op = torch.unsqueeze(conv_op, 0)

    LPLayer = LapLayer(conv_op)
    """

    Unet_chi = Unet(4, 1, 1)
    #Unet_lfs = Unet(4, 1, 1)

    print(Unet_chi.state_dict)
    print(get_parameter_number(Unet_chi))
    ###### use this line to check if all layers 
    ###### are leanrable in this programe. 
    ## train network
    TrainNet(Unet_chi, LR = 0.001, Batchsize = 32, Epoches = 100 , useGPU = True)

