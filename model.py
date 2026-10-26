import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
# Multi Scale Encoder
#
# 输入:
# x:
# (B,1,512)
#
# 输出:
# (B,64,512)
#
# ============================================================


class MultiScaleEncoder(nn.Module):

    def __init__(self):

        super().__init__()


        # 小尺度卷积
        self.conv3=nn.Conv1d(
            1,
            16,
            kernel_size=3,
            padding=1
        )


        # 中尺度卷积
        self.conv7=nn.Conv1d(
            1,
            16,
            kernel_size=7,
            padding=3
        )


        # 大尺度卷积
        self.conv15=nn.Conv1d(
            1,
            16,
            kernel_size=15,
            padding=7
        )


        # 特征融合

        self.fusion=nn.Conv1d(
            48,
            64,
            kernel_size=1
        )



    def forward(self,x):


        # x:
        # (B,1,512)


        x1=F.relu(
            self.conv3(x)
        )


        # (B,16,512)


        x2=F.relu(
            self.conv7(x)
        )


        # (B,16,512)



        x3=F.relu(
            self.conv15(x)
        )


        # (B,16,512)



        x=torch.cat(
            [
                x1,
                x2,
                x3
            ],
            dim=1
        )


        # (B,48,512)



        x=self.fusion(x)


        # (B,64,512)


        return x







# ============================================================
# Innovation 1:
#
# Dynamic Frequency Routing Module
#
# 动态频率路由
#
# 输入:
# (B,64,512)
#
# 输出:
# (B,64,512)
#
# ============================================================


class DynamicFrequencyRouting(nn.Module):


    def __init__(self):

        super().__init__()



        self.mlp=nn.Sequential(

            nn.Linear(
                257,
                128
            ),

            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(
                128,
                257
            ),

            nn.Sigmoid()

        )



    def forward(self,x):


        B,C,L=x.shape



        # FFT
        #
        # (B,64,257)

        fft=torch.fft.rfft(
            x,
            dim=-1
        )
        

        magnitude=torch.abs(
            fft
        )



        # 通道平均

        freq_feature=magnitude.mean(
            dim=1
        )


        # (B,257)



        # 学习频率权重

        weight=self.mlp(
            freq_feature
        )


        # (B,257)


        weight=weight.unsqueeze(1)


        # (B,1,257)



        # 频率重标定

        magnitude=magnitude*weight



        phase=torch.angle(
            fft
        )



        new_fft=torch.polar(
            magnitude,
            phase
        )



        out=torch.fft.irfft(
            new_fft,
            n=L,
            dim=-1
        )



        return out







# ============================================================
# Innovation 2:
#
# Noise Aware Temporal Consistency Module
#
# 噪声感知时间一致性模块
#
# 输入:
# (B,64,512)
#
# 输出:
# (B,64,512)
#
# ============================================================



class NoiseAwareTemporal(nn.Module):


    def __init__(self):

        super().__init__()



        # 噪声概率估计

        self.noise_mask=nn.Sequential(

            nn.Conv1d(
                64,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Conv1d(
                64,
                64,
                1
            ),

            nn.Sigmoid()

        )



        # 时间一致性建模

        self.temporal=nn.Sequential(

            nn.Conv1d(
                64,
                64,
                5,
                padding=2
            ),

            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Conv1d(
                64,
                64,
                3,
                padding=1
            )

        )



    def forward(self,x):


        # 预测噪声区域

        mask=self.noise_mask(x)


        # 抑制噪声

        x_clean=x*(1-mask)



        # 时间建模

        out=self.temporal(
            x_clean
        )


        # residual

        out=out+x


        return out







# ============================================================
# Decoder
#
# (B,64,512)
#
# ->
#
# (B,512)
#
# ============================================================


class Decoder(nn.Module):


    def __init__(self):

        super().__init__()



        self.net=nn.Sequential(

            nn.Conv1d(
                64,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),


            nn.Conv1d(
                32,
                1,
                1
            )

        )



    def forward(self,x):


        return self.net(x)








# ============================================================
# ANFINet
#
# ============================================================


class ANFINet(nn.Module):


    def __init__(self):

        super().__init__()


        self.encoder=MultiScaleEncoder()


        self.frequency=DynamicFrequencyRouting()


        self.temporal=NoiseAwareTemporal()


        self.decoder=Decoder()



    def forward(self,x):


        # 输入:
        # (B,512)


        x=x.unsqueeze(1)


        # (B,1,512)



        feature=self.encoder(x)


        # (B,64,512)



        feature=self.frequency(
            feature
        )


        # (B,64,512)



        feature=self.temporal(
            feature
        )


        # (B,64,512)



        out=self.decoder(
            feature
        )


        # (B,1,512)



        out=out.squeeze(1)


        # (B,512)


        return out
    
    
class ANFINet_Baseline(nn.Module):


    def __init__(self):
    
        super().__init__()


        self.encoder=MultiScaleEncoder()


        self.frequency=DynamicFrequencyRouting()


        self.temporal=NoiseAwareTemporal()


        self.decoder=Decoder()



    def forward(self,x):


        # 输入:
        # (B,512)


        x=x.unsqueeze(1)


        # (B,1,512)



        feature=self.encoder(x)


        # (B,64,512)

        out=self.decoder(
            feature
        )


        # (B,1,512)



        out=out.squeeze(1)


        # (B,512)


        return out



# ==========================================================
# 消融模型2:
# w/o DFRM
# 去掉动态频率路由模块
# ==========================================================


class ANFINet_WO_DFRM(nn.Module):


    def __init__(self):
    
        super().__init__()
    
    
        self.encoder=MultiScaleEncoder()


        self.frequency=DynamicFrequencyRouting()


        self.temporal=NoiseAwareTemporal()


        self.decoder=Decoder()



    def forward(self,x):


        # 输入:
        # (B,512)


        x=x.unsqueeze(1)


        # (B,1,512)



        feature=self.encoder(x)


        # (B,64,512)


        feature=self.temporal(
            feature
        )


        # (B,64,512)



        out=self.decoder(
            feature
        )


        # (B,1,512)



        out=out.squeeze(1)


        # (B,512)


        return out




# ==========================================================
# 消融模型3:
# w/o NTCM
# 去掉噪声感知时间一致性模块
# ==========================================================


class ANFINet_WO_NTCM(nn.Module):


    def __init__(self):
   
        super().__init__()
        self.encoder=MultiScaleEncoder()
        self.frequency=DynamicFrequencyRouting()
        self.temporal=NoiseAwareTemporal()
        self.decoder=Decoder()

    def forward(self,x):
        # 输入:
        # (B,512)
        x=x.unsqueeze(1)
        # (B,1,512)
        feature=self.encoder(x)
        # (B,64,512)
        feature=self.frequency(
            feature
        )
        # (B,64,512)
        out=self.decoder(
            feature
        )
        # (B,1,512)
        out=out.squeeze(1)
        # (B,512)
        return out