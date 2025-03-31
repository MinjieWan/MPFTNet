import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stx
import math
from functools import partial
from functools import reduce

##########################################################################
##---------- Channel Feature Selection Fusion (CFSF) ----------

class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        return sig_pool

class CFSF(nn.Module):
    def __init__(self, in_channels, height=2,reduction=8,bias=False):
        super(CFSF, self).__init__()
        
        self.height = height
        self.in_channels = in_channels
        self.conv_tran = nn.Sequential(
            nn.Conv2d(self.height * self.in_channels, self.in_channels, 1, padding=0, bias=bias),
            nn.LeakyReLU(0.2),
            )

        self.ca = ChannelAttention(self.in_channels, reduction_ratio = reduction)
        self.conv_divide = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * self.height, 1, padding=0, bias=bias),
        )        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]

        diffY = inp_feats[1].size()[2] - inp_feats[0].size()[2]
        diffX = inp_feats[1].size()[3] - inp_feats[0].size()[3]
        if diffY > 0 or diffX > 0:
            start_y = max(diffY // 2, 0)
            end_y = start_y + inp_feats[0].size()[2]
            start_x = max(diffX // 2, 0)
            end_x = start_x + inp_feats[0].size()[3]
            inp_feats[1] = inp_feats[1][:, :, start_y:end_y, start_x:end_x]

        inp_feats = torch.cat((inp_feats[0], inp_feats[1]), dim=1)
        inp_feats_0 = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        inp_feats = self.conv_tran(inp_feats)
        ca_feats = self.ca(inp_feats)
        
        out_feats = self.conv_divide(ca_feats)
        out_feats = out_feats.reshape(batch_size, self.height, n_feats, 1, 1) 
        out_feats = self.softmax(out_feats)
        feats_V = torch.sum(inp_feats_0*out_feats, dim=1)

        return feats_V


##########################################################################
##---------- Non-local Multiscale Pooling (NMP) ----------

class MP(nn.ModuleList):
    def __init__(self, pool_sizes=[1,2,4,8]):
        super(MP, self).__init__()
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size)
                )
            )
    def forward(self, x):
        out = []
        b, c, _, _ = x.size()
        for index, module in enumerate(self):
            out.append(module(x))
        return torch.cat([output.view(b, c, -1) for output in out], -1)
    
class NLBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, pool_sizes=[1,2,4,8]):
        super(NLBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.Conv_query = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, 1),
            nn.LeakyReLU(0.2)
        )
        self.Conv_key = self.Conv_query
        self.Conv_value = nn.Conv2d(self.in_channels, self.value_channels, 1)

        self.ConvOut = nn.Conv2d(self.value_channels, self.out_channels, 1)
        self.mp = MP(pool_sizes)
        nn.init.constant_(self.ConvOut.weight, 0)
        nn.init.constant_(self.ConvOut.bias, 0)

    def forward(self, x):

        b, c, h, w = x.size()

        value = self.mp(self.Conv_value(x)).permute(0, 2, 1)
        key = self.mp(self.Conv_key(x))
        query = self.Conv_query(x).view(b, self.key_channels, -1).permute(0, 2, 1)

        Concat_QK = torch.matmul(query, key)
        Concat_QK = (self.key_channels ** -.5) * Concat_QK
        Concat_QK = F.softmax(Concat_QK, dim=-1)

        # Aggregate_QKV = [batch, h*w, Value_channels]
        Aggregate_QKV = torch.matmul(Concat_QK, value)
        # Aggregate_QKV = [batch, value_channels, h*w]
        Aggregate_QKV = Aggregate_QKV.permute(0, 2, 1).contiguous()
        # Aggregate_QKV = [batch, value_channels, h*w] -> [batch, value_channels, h, w]
        Aggregate_QKV = Aggregate_QKV.view(b, self.value_channels, *x.size()[2:])
        # Conv out
        Aggregate_QKV = self.ConvOut(Aggregate_QKV)

        out = Aggregate_QKV + x

        return out

class NMP(nn.Module):
    def __init__(self, n_feat, bias=False, groups=1):
        super(NMP, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act, 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
        )

        self.act = act

        self.nlblock = NLBlock(in_channels=n_feat, key_channels=n_feat // 2, value_channels=n_feat)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.nlblock(res))
        res = x + res
        return res


##########################################################################
##---------- Resizing Modules ----------    
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class US(nn.Module):
    def __init__(self, in_channels):
        super(US, self).__init__()
        self.up = UpSample(in_channels,2,1)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        if diffY > 0 or diffX > 0:
            start_y = max(diffY // 2, 0)
            end_y = start_y + x2.size()[2]
            start_x = max(diffX // 2, 0)
            end_x = start_x + x2.size()[3]
            x1 = x1[:, :, start_y:end_y, start_x:end_x]

        x = x2 + x1
        return x


##########################################################################
##---------- Multi-resolution Global Feature Integration (MGFI) ----------
class MGFI(nn.Module):
    def __init__(self, n_feat, chan_factor, bias,groups):
        super(MGFI, self).__init__()

        self.n_feat = n_feat

        self.nmp_top = NMP(int(n_feat*chan_factor**0), bias=bias, groups=groups)
        self.nmp_mid = NMP(int(n_feat*chan_factor**1), bias=bias, groups=groups)
        self.nmp_bot = NMP(int(n_feat*chan_factor**2), bias=bias, groups=groups)

        self.down2 = DownSample(int((chan_factor**0)*n_feat),2,chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor**0)*n_feat),2,chan_factor), 
            DownSample(int((chan_factor**1)*n_feat),2,chan_factor)
        )

        self.up21 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up32 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        self.cfsf_top = CFSF(int(n_feat*chan_factor**0), 2)
        self.cfsf_mid = CFSF(int(n_feat*chan_factor**1), 2)

    def forward(self, x):
        x_top = x.clone()
        x_mid = self.down2(x)
        x_bot = self.down4(x)

        x_top = self.nmp_top(x_top)
        x_top = self.cfsf_top([x_top, self.up21(x_mid)])
        x_mid = self.nmp_mid(x_mid)
        x_mid = self.cfsf_mid([x_mid, self.up32(x_bot)])
        x_bot = self.nmp_bot(x_bot)

        x_top = self.nmp_top(x_top)
        x_top = self.cfsf_top([x_top, self.up21(x_mid)])
        x_mid = self.nmp_mid(x_mid)
        x_mid = self.cfsf_mid([x_mid, self.up32(x_bot)])
        x_bot = self.nmp_bot(x_bot)

        x_top = self.nmp_top(x_top)
        x_top = self.cfsf_top([x_top, self.up21(x_mid)])
        x_mid = self.nmp_mid(x_mid)
        x_mid = self.cfsf_mid([x_mid, self.up32(x_bot)])

        x_top = self.nmp_top(x_top)
        x_top = self.cfsf_top([x_top, self.up21(x_mid)])

        out = self.conv_out(x_top)
        out = out + x

        return out

##########################################################################
##---------- MPFTNet  -----------------------
class MPFTNet(nn.Module):
    def __init__(self,
        inp_channels=1,
        out_channels=1,
        n_feat=64,
        chan_factor=2,
        bias=False,
        task= None
    ):
        super(MPFTNet, self).__init__()

        self.task = task

        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)

        self.mgfi1 = MGFI(int(n_feat), chan_factor, bias, groups=1)
        self.mgfi2 = MGFI(int(n_feat), chan_factor, bias, groups=2)
        self.mgfi3 = MGFI(int(n_feat), chan_factor, bias, groups=4)

        self.down = DownSample(int(n_feat),2,1)
        self.up = US(int(n_feat))

        self.conv_out = nn.Conv2d(int(n_feat), out_channels, kernel_size=3, padding=1, bias=bias)
        

    def forward(self, inp_img):

        en_feats_1 = self.conv_in(inp_img)

        en_feats_2 = self.mgfi1(en_feats_1)

        down_feats_1 = self.down(en_feats_2)
        en_feats_3 = self.mgfi2(down_feats_1)

        down_feats_2 = self.down(en_feats_3)
        en_feats_4 = self.mgfi3(down_feats_2)

        de_feats_4 = self.mgfi3(en_feats_4)

        up_feats_2 = self.up(de_feats_4, en_feats_3)
        de_feats_3 = self.mgfi2(up_feats_2)

        up_feats_1 = self.up(de_feats_3, en_feats_2)
        de_feats_2 = self.mgfi1(up_feats_1)
       
        out_img = self.conv_out(de_feats_2)
        
        out_img += inp_img

        return out_img
