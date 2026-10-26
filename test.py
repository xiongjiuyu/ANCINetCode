import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
from time import time
import os
#import zhibiao
import matplotlib.pyplot as plt
from model import ANFINet,ANFINet_Baseline,ANFINet_WO_DFRM,ANFINet_WO_NTCM

BATCH_SIZE = 50
# datadir = "data/test"
datadir = "hdata"
def get_data(data="mix"):
    if data not in ["eog","emg","mix"]:
        print("请输入正确的伪迹类型")
        exit()
    elif data == "eog":
        verify_input = torch.from_numpy(np.load(datadir + '/input_EOG.npy'))
        verify_output = torch.from_numpy(np.load(datadir + '/clean_EOG.npy'))
    elif data == "emg":
        verify_input = torch.from_numpy(np.load(datadir + '/input_EMG.npy'))
        verify_output = torch.from_numpy(np.load(datadir + '/clean_EMG.npy'))
    elif data == "mix":
        verify_input = torch.from_numpy(np.load(datadir + '/input_MIX.npy'))
        verify_output = torch.from_numpy(np.load(datadir + '/clean_MIX.npy'))
    
    
    test_torch_dataset = Data.TensorDataset(verify_input, verify_output)
    
    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return test_loader




def test(model, device, test_loader,num,num_x,input_z ,output_z,pre_z):
    model.eval()
    step_num=0
    loss_epoh=0
    #for batch_idx, (test_input, indicator, test_output) in enumerate(test_loader):
    for batch_idx, (test_input, test_output) in enumerate(test_loader):
        #indicator = indicator.float().to(device)
        test_input=test_input.float().to(device)
        test_output=test_output.float().to(device)
        output = model(test_input)
        output = output.detach().cpu()
        test_output = test_output.detach().cpu()
        test_input = test_input.detach().cpu()      
        
        input_z[step_num*BATCH_SIZE:(step_num+1)*BATCH_SIZE] = test_input
        
        output_z[step_num*BATCH_SIZE:(step_num+1)*BATCH_SIZE] = test_output
        
        pre_z[step_num*BATCH_SIZE:(step_num+1)*BATCH_SIZE] = output
        step_num +=1


def test_model(modelname="ANFINet", noisetype = "mix"):
    if noisetype not in ["eog","emg","mix"]:
        print("请输入正确的伪迹名称")
        exit()
    if modelname not in ["ANFINet","ANFINet_Baseline","ANFINet_WO_DFRM","ANFINet_WO_NTCM"]:
        print("请输入正确的模型名称")
        exit()
    if modelname == "ANFINet":
        model=ANFINet()
    elif modelname == "ANFINet_Baseline":
        model=ANFINet_Baseline()
    elif modelname == "ANFINet_WO_DFRM":
        model=ANFINet_WO_DFRM()
    elif modelname == "ANFINet_WO_NTCM":
        model=ANFINet_WO_NTCM()
    test_loader = get_data(data=noisetype)
    model_name = modelname + '_' + noisetype +'.pth'

    print("torch.cuda.is_available() = ", torch.cuda.is_available())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    begin_time = time()
    if os.path.exists(model_name):
        print('load model')
        model.load_state_dict(torch.load(model_name))
    num = len(test_loader.dataset)
    num_x = num//10
    test_input_z = torch.ones(num,512)
    test_output_z = torch.ones(num,512)  
    pre_z = torch.ones(num,512)
    test(model, device, test_loader,num,num/10,test_input_z ,test_output_z ,pre_z )

    #保存预测结果到pre.npy
    numpy_array = test_output_z.numpy()
    np.save(datadir + '/' + modelname + '_' + noisetype +'_pre.npy', pre_z)

    
###############测试模型#############
#modelname有以下选项
#   ANFINet,
#   ANFINet_Baseline,
#   ANFINet_WO_DFRM,
#   ANFINet_WO_NTCM
##################################
# test_model(modelname="ANFINet",noisetype="eog")
# test_model(modelname="ANFINet",noisetype="emg")
# test_model(modelname="ANFINet",noisetype="mix")
test_model(modelname="ANFINet_Baseline")
test_model(modelname="ANFINet_WO_DFRM")
test_model(modelname="ANFINet_WO_NTCM")