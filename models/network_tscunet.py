# -*- coding: utf-8 -*-
import math
import re
import torch
import torch.nn as nn
import numpy as np

from einops.einops import rearrange
from models.network_scunet import ConvTransBlock, RRDBUpsample, Upconv


class TSCUNetBlock(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(TSCUNetBlock, self).__init__()

        self.head_dim = 32
        self.window_size = 8
        self.config = config
        self.dim = dim

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]
        
        begin = 0
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[0])] + \
                      [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[1])] + \
                      [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[2])] + \
                      [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [Upconv(8*dim, 4*dim, 2, 2, bias=False),] + \
                      [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[4])]
                      
        begin += config[4]
        self.m_up2 = [Upconv(4*dim, 2*dim, 2, 2, bias=False),] + \
                      [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[5])]
                      
        begin += config[5]
        self.m_up1 = [Upconv(2*dim, dim, 2, 2, bias=False),] + \
                    [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[6])]

        self.m_res = [nn.Conv2d(dim, dim, 3, 1, 1, bias=False)]
        self.m_tail = [nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False), nn.LeakyReLU(0.2, True)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)

        self.m_res = nn.Sequential(*self.m_res)
        self.m_tail = nn.Sequential(*self.m_tail)

    def forward(self, x0):
        x1 = self.m_head(x0)

        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)

        x = x + self.m_res(x1)
        x = self.m_tail(x)
        
        return x


class TSCUNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, clip_size=5, nb=2, dim=64, drop_path_rate=0.0, input_resolution=256, scale=2, state=None):
        super(TSCUNet, self).__init__()

        if state:
            in_nc =  state['m_head.0.weight'].shape[1]
            dim =    state['m_head.0.weight'].shape[0]
            out_nc = state['m_tail.0.weight'].shape[0]

            clip_size = len([k for k in state.keys() if re.match(re.compile(f'm_layers\..\.m_body\.0\.trans_block\.mlp\.0\.weight'), k)]) * 2 + 1
            nb = len([k for k in state.keys() if re.match(re.compile(f'm_layers\.0\.m_body\..\.trans_block\.mlp\.0\.weight'), k)])

            scale = 2 ** max(0, len([k for k in state.keys() if re.match(re.compile('m_upsample\.0\.up\.[0-9]+\.weight'), k)])-1)
            input_resolution = 64 if scale > 1 else 256 

        if clip_size % 2 == 0:
            raise ValueError('TSCUNet clip_size must be odd')

        self.clip_size = clip_size
        self.dim = dim
        self.scale = scale
        
        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]
        self.m_tail = [nn.Conv2d(dim, out_nc, 3, 1, 1, bias=False)]

        self.m_layers = [TSCUNetBlock(dim * 3, dim, [nb] * 7, dim, drop_path_rate, input_resolution) for _ in range((clip_size-1)//2)]

        if self.scale > 1:
            self.m_upsample = [RRDBUpsample(dim, nb=2, scale=self.scale)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_tail = nn.Sequential(*self.m_tail)
        self.m_layers = nn.ModuleList(self.m_layers)

        if scale > 1:
            self.m_upsample = nn.Sequential(*self.m_upsample)

        if state:
            self.load_state_dict(state, strict=True)

    def forward(self, x):
        b, t, c, h, w = x.size()
        if t != self.clip_size:
            raise ValueError(f'input clip size {t} does not match model clip size {self.clip_size}')

        paddingH = int(np.ceil(h/64)*64-h)
        paddingW = int(np.ceil(w/64)*64-w)
        if not self.training:
            paddingH += 64
            paddingW += 64

        paddingLeft = paddingW // 2
        paddingRight = paddingW // 2
        paddingTop = paddingH // 2
        paddingBottom = paddingH // 2

        x = nn.ReflectionPad2d((paddingLeft, paddingRight, paddingTop, paddingBottom))(x.view(-1, c, h, w)).view(b, -1, c, h + paddingH, w + paddingW)

        for layer in self.m_layers:
            temp = [None] * (t - 2)
        
            for i in range(t - 2):
                window = x[:, i:i+3, ...]
                if window.size(2) != self.dim:
                    window = self.m_head(window.view(-1, c, h + paddingH, w + paddingW)).view(b, -1, self.dim, h + paddingH, w + paddingW)

                temp[i] = layer(window.view(b, -1, h + paddingH, w + paddingW))

            x = torch.stack(temp, dim=1)
            t = x.size(1)

        x = x.squeeze(1)
        
        if self.scale > 1:
            x = self.m_upsample(x)
        x = self.m_tail(x)

        x = x[..., paddingTop*self.scale:paddingTop*self.scale+h*self.scale, paddingLeft*self.scale:paddingLeft*self.scale+w*self.scale]
        return x



if __name__ == '__main__':
    # torch.cuda.empty_cache()
    scale = 2
    batch = 3
    clip_size = 5
    crop_size = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TSCUNet(clip_size=clip_size).to(device)
    net = TSCUNet(clip_size=clip_size, state=net.state_dict()).to(device)

    x = torch.randn(batch, clip_size, 3, crop_size // scale, crop_size // scale).to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    net.train()
    #with torch.no_grad():
    with torch.cuda.amp.autocast():
        y = net(x)

    end.record()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)

    effective_frames = ((crop_size * crop_size) / (1440 * 1080)) * batch

    print(f'Time: {total_time}ms')
    print(f'FPS: {effective_frames / (total_time / 1000)}')
    print(f'Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
    print(f'Model size: {sum(p.numel() * p.element_size() for p in net.parameters() if p.requires_grad) / 1024 / 1024} MB')
