################### train DCRNet #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from TrainingDataLoader import *
import sys
sys.path.append('./Model/')
from DCRNet import * 
 
#########  Section 1: DataSet Load #############
def DataLoad(Batch_size):
    DATA_DIRECTORY = '../'
    DATA_LIST_PATH = './test_IDs.txt'
    ## get mask;
    matImage = scio.loadmat('Real_Mask_Acc4_forTraining.mat')
    mask = matImage['mask']  
    mask = np.array(mask)
    
    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH, mask)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def SaveNet(dcrnet, epo, enSave = False):
    print('save results')
    #### save the
    if enSave:
        pass
    else:
        torch.save(dcrnet.state_dict(), './DCRNet_AF4_final.pth')
        torch.save(dcrnet.state_dict(), ("DCRNet_AF4_%s.pth" % epo))

def TrainNet(dcrnet, LR = 0.001, Batchsize = 32, Epoches = 100 , useGPU = True):
    print('DeepResNet')
    print('DataLoad')
    trainloader = DataLoad(Batchsize)
    print('Dataload Ends')

    print('Training Begins')
    criterion = nn.MSELoss(size_average=None, reduce=None, reduction='sum')
    criterion2 = nn.L1Loss(reduction='sum')
    optimizer2 = optim.Adam(dcrnet.parameters())
    scheduler2 = LS.MultiStepLR(optimizer2, milestones = [40, 80], gamma = 0.1)

    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dcrnet = nn.DataParallel(dcrnet)
            dcrnet.to(device)

            for epoch in range(1, Epoches + 1):
                
                if epoch % 20 == 0:
                    SaveNet(dcrnet, epoch, enSave = False)

                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    image_r, image_i, label_r, label_i, k0_r, k0_i, mask, Name = data
                    image_r = image_r.to(device)
                    image_i = image_i.to(device)
                    label_r = label_r.to(device)
                    label_i = label_i.to(device)

                    k0_r = k0_r.to(device)
                    k0_i = k0_i.to(device)
                    mask = mask.to(device)

                    ###print(image_i.size())
                    ## zero the gradient buffers 
                    optimizer2.zero_grad()
                    ## forward: 
                    ini_r, ini_i, pred_r, pred_i = dcrnet(image_r, image_i, k0_r, k0_i, mask)

                    loss1 = criterion(pred_r, label_r)
                    loss2 = criterion(pred_i, label_i)

                    loss3 = criterion(ini_r, label_r)
                    loss4 = criterion(ini_i, label_i)

                    loss = loss1 + loss2 
                    loss5 = loss3 + loss4

                    loss5.backward(retain_graph = True)
                    loss.backward()
                    ##
                    optimizer2.step()
                    optimizer2.zero_grad()
                    
                    ## print statistical information 
                    ## print every 20 mini-batch size
                    if i % 20 == 0:
                        acc_loss1 = loss.item()   
                        acc_loss2 = loss5.item()
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss_ini: %f, Loss_final: %f \n lr2: %f, used time: %d s' %
                            (epoch, i + 1, acc_loss2, acc_loss1, optimizer2.param_groups[0]['lr'], time_end - time_start))   
                scheduler2.step()
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')
    SaveNet(dcrnet, Epoches, enSave = False)

if __name__ == '__main__':
    ## data load
    ## create network 
    dcrnet = DCRNet(5)
    dcrnet.apply(weights_init)
    print(dcrnet.state_dict)
    print(get_parameter_number(dcrnet))
    ## train network
    TrainNet(dcrnet, LR = 0.001, Batchsize = 32, Epoches = 100, useGPU = True)

