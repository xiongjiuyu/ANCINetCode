import torch
import torch.nn.functional as F
import time
from model import ANFINet,ANFINet_Baseline,ANFINet_WO_DFRM,ANFINet_WO_NTCM
import numpy as np
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
BATCH_SIZE = 50
def get_data(data = "mix"):
    if data not in ["eog","emg","mix"]:
        print("请输入正确的伪迹类型")
        exit()
    elif data == "eog":
        train_input = torch.from_numpy(np.load('data/train/input_EOG.npy'))
        train_output = torch.from_numpy(np.load('data/train/clean_EOG.npy'))
        verify_input = torch.from_numpy(np.load('data/val/input_EOG.npy'))
        verify_output = torch.from_numpy(np.load('data/val/clean_EOG.npy'))
    elif data == "emg":
        train_input = torch.from_numpy(np.load('data/train/input_EMG.npy'))
        train_output = torch.from_numpy(np.load('data/train/clean_EMG.npy'))
        verify_input = torch.from_numpy(np.load('data/val/input_EMG.npy'))
        verify_output = torch.from_numpy(np.load('data/val/clean_EMG.npy'))
    elif data == "mix":
        train_input = torch.from_numpy(np.load('data/train/input_MIX.npy'))
        train_output = torch.from_numpy(np.load('data/train/clean_MIX.npy'))
        verify_input = torch.from_numpy(np.load('data/val/input_MIX.npy'))
        verify_output = torch.from_numpy(np.load('data/val/clean_MIX.npy'))

    train_torch_dataset = Data.TensorDataset(train_input, train_output)
    
    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    
    
    verify_torch_dataset = Data.TensorDataset(verify_input, verify_output)
    
    verify_loader = Data.DataLoader(
        dataset=verify_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    return train_loader,verify_loader






# ==================================================
# Log Spectrum Loss
#
# 频域损失
#
# ==================================================


def freq_loss(
        pred,
        target
):


    pred_fft=torch.fft.rfft(
        pred,
        dim=-1
    )


    target_fft=torch.fft.rfft(
        target,
        dim=-1
    )



    pred_mag=torch.abs(
        pred_fft
    )


    target_mag=torch.abs(
        target_fft
    )



    pred_log=torch.log1p(
        pred_mag
    )


    target_log=torch.log1p(
        target_mag
    )



    return F.mse_loss(
        pred_log,
        target_log
    )








# ==================================================
# Temporal Loss
#
# 时间一致性
#
# ==================================================


def temporal_loss(
        pred,
        target
):


    dp=pred[:,1:]-pred[:,:-1]


    dt=target[:,1:]-target[:,:-1]



    return F.mse_loss(
        dp,
        dt
    )









# ==================================================
# Total Loss
#
# MSE
# +
# Log Spectrum
# +
# Temporal
#
# ==================================================

def total_loss(
        pred,
        target
):


    mse=F.mse_loss(
        pred,
        target
    )


    lf=freq_loss(
        pred,
        target
    )


    lt=temporal_loss(
        pred,
        target
    )



    loss=(

        mse

        +

        0.02*lf

        +

        0.05*lt

    )


    return loss,mse,lf,lt

def verify(model, device, verify_loader):
    model.eval()
    step_num=0
    loss_epoh=0
    for batch_idx, (verify_input, verify_output) in enumerate(verify_loader):        
        step_num +=1
        verify_input=verify_input.float().to(device)
        verify_output=verify_output.float().to(device)
        output = model(verify_input)
        loss,mse,lf,lt=total_loss(output,verify_output)
        loss_epoh+=loss.item()

    print(loss_epoh/step_num)
    return loss_epoh/step_num





# ==================================================
# Training
# ==================================================

#
batch_size=100
epochs=150
lr=5e-4
loss_f = nn.MSELoss(reduction='mean')




def train_model(modelname="ANFINet", noisetype="mix"):
    start = time.time()
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
    print(f"will train model:{modelname}")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer=torch.optim.Adam(

        model.parameters(),

        lr=lr

    )
    trainlaoder,valloader = get_data(data=noisetype)
    loss_best = 10000
    cout = 10
    for epoch in range(epochs):


        model.train()

        
        for noisy,clean in trainlaoder:
            noisy=noisy.float().to(device)  
            clean=clean.float().to(device)
    

            pred=model(
                noisy
            )


            loss,mse,lf,lt=total_loss(

                pred,

                clean

            )



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
        loss_val = verify(model, device, valloader)
        print(f"val loss = {loss_val}")
        if loss_best > loss_val:
            loss_best = loss_val
            torch.save(model.state_dict(),modelname + "_" + noisetype +".pth")
            print("save model")
            cout =10
        cout -= 1
        if (cout == 0):
            print("########################训练提前结束###################################")
            break
        print(

            f"Epoch:{epoch+1}/{epochs} "

            f"Loss:{loss.item():.4f} "

            f"MSE:{mse.item():.4f} "

            f"Freq:{lf.item():.4f} "

            f"Time:{lt.item():.4f}"

        )
    print("Finished!")
    end = time.time()
    elapsed = end - start
    print(f"运行时间: {elapsed:.4f} 秒") 



###############训练模型#############
#modelname有以下选项
#   ANFINet,
#   ANFINet_Baseline,
#   ANFINet_WO_DFRM,
#   ANFINet_WO_NTCM
##################################
train_model(modelname="ANFINet",noisetype="eog")
# train_model(modelname="ANFINet",noisetype="emg")
# train_model(modelname="ANFINet",noisetype="mix")
# train_model(modelname="ANFINet_Baseline")
# train_model(modelname="ANFINet_WO_DFRM")
# train_model(modelname="ANFINet_WO_NTCM")

