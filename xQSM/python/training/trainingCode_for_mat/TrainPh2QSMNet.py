################### train AutoBCS framework #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from xQSM import * 
from TrainingDataLoad import * 
#########  Section 1: DataSet Load #############
def DataLoad(Batch_size):
    DATA_DIRECTORY = '..'
    DATA_LIST_PATH = './test_IDs.txt'
    
    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def SaveNet(Chi_Net, epo, enSave = False):
    print('save results')
    #### save the
    if enSave:
        pass
    else:
        torch.save(Chi_Net.state_dict(), './ChiNet_Latest.pth')
        torch.save(Chi_Net.state_dict(), ("./ChiNet_%s.pth" % epo))

def TrainNet(Chi_Net, LR = 0.001, Batchsize = 32, Epoches = 100 , useGPU = True):
    print('DataLoader setting begins')
    trainloader = DataLoad(Batchsize)
    print('Dataloader settting end')

    print('Training Begins')
    criterion = nn.MSELoss(reduction='sum')

    optimizer1 = optim.Adam(Chi_Net.parameters())

    scheduler1 = LS.MultiStepLR(optimizer1, milestones = [50, 80], gamma = 0.1)
    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            Chi_Net = nn.DataParallel(Chi_Net)
            Chi_Net.to(device)

            for epoch in range(1, Epoches + 1):

                if epoch % 20 == 0:
                    SaveNet(Chi_Net, epoch, enSave = False)

                acc_loss = 0.0
                for i, data in enumerate(trainloader):
                    lfss, chis, name = data
                    lfss = lfss.to(device)
                    chis = chis.to(device)
                    ## zero the gradient buffers 
                    optimizer1.zero_grad()
                    ## forward: 
                    pred_chis = Chi_Net(lfss)
                    ## loss
                    loss1 = criterion(pred_chis, chis)
                    ## backward
                    loss1.backward()
                    ##
                    optimizer1.step()

                    optimizer1.zero_grad()
                    ## print statistical information 
                    ## print every 20 mini-batch size
                    if i % 19 == 0:
                        acc_loss1 = loss1.item()   
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss1: %f, lr1: %f,  used time: %d s' %
                            (epoch, i + 1, acc_loss1, optimizer1.param_groups[0]['lr'], time_end - time_start))  
                scheduler1.step()
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')
    SaveNet(Chi_Net, Epoches)

if __name__ == '__main__':
    ## load laplacian operator; 

    Chi_Net = xQSM(2, 64)
    Chi_Net.apply(weights_init)
    Chi_Net.train()

    print(Chi_Net.state_dict)
    print(get_parameter_number(Chi_Net))

    ## train network
    TrainNet(Chi_Net, LR = 0.001, Batchsize = 32, Epoches = 100 , useGPU = True)

