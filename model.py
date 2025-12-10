import random

import torch
import math
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.autograd as ag
# GPU = 1
class FE_HSI(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim1,act,params):
        super().__init__()
        self.params=params
        self.output_dim=output_dim

        self.spa_fe=FE( input_dim=input_dim,output_dim=output_dim,hidden_dim=hidden_dim1,act=act,params=params)





    def forward(self, x):


        x=x.reshape(-1,x.shape[-1])


        x=self.spa_fe(x).reshape(-1,self.params['patches']**2,self.output_dim )





        x=torch.mean(x,1)


        return x
class FE_LiDAR(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim1,act,params):
        super().__init__()
        self.params=params
        self.output_dim=output_dim

        self.spe_fe = FE(input_dim=params['patches']**2, output_dim=output_dim, hidden_dim=hidden_dim1, act=act, params=params)




    def forward(self, x):
        a=x.shape[0]


        x=x.reshape(-1,x.shape[-1])
        x=x.reshape(-1,self.params['patches']**2,x.shape[-1]).permute(0,2,1).reshape(-1,self.params['patches']**2)


        x = torch.mean(self.spe_fe(x).reshape(a,-1, self.output_dim),1)







        return x
class FE(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim,act,params):
        super().__init__()
        self.params=params
        self.input_dim=input_dim
        self.output_dim = output_dim
        self.hiddent_dim=hidden_dim
        self.act=act
        self.layer=nn.Sequential()
        if len(self.hiddent_dim)==0:
            self.layer.append(nn.BatchNorm1d(self.input_dim))


            self.layer.append(nn.Linear(self.input_dim, output_dim))
            self.layer.append(self.act)
        else:
            for i in range(len(hidden_dim)):
                if i==0:
                    self.layer.append(nn.BatchNorm1d(self.input_dim))

                    self.layer.append(nn.Dropout(0.2))


                    self.layer.append(nn.Linear(self.input_dim, self.hiddent_dim[0]))

                    self.layer.append(self.act)
                if i!=0 and i!=len(hidden_dim)-1:
                    self.layer.append(nn.BatchNorm1d(self.hiddent_dim[i-1]))
                    self.layer.append(nn.Dropout(0.2))


                    self.layer.append(nn.Linear(self.hiddent_dim[i-1], self.hiddent_dim[i]))

                    self.layer.append(self.act)
                if i==len(hidden_dim)-1:
                    self.layer.append(nn.BatchNorm1d(self.hiddent_dim[i - 1]))
                    self.layer.append(nn.Dropout(0.2))


                    self.layer.append(nn.Linear(self.hiddent_dim[i-1], self.output_dim))

                    self.layer.append(self.act)




    def forward(self, x):

        x=x.reshape(-1,self.input_dim)

        x=self.layer(x)






        return x



class C_Encoder (nn.Module):
    def __init__(self, dim,e_num,euc_num,d_num):
        super().__init__()
        self.e_num=e_num
        self.euc_num=euc_num
        self.d_num=d_num





        self.dim = dim
        self.layer1 = nn.Sequential(nn.BatchNorm1d(dim), nn.Linear(dim, e_num[0]), nn.Tanh())
        for i in range(len(self.e_num) -2):

            self.layer1.append(nn.BatchNorm1d(self.e_num[i]),)

            self.layer1.append( nn.Linear(self.e_num[i], self.e_num[i + 1]))
            self.layer1.append(nn.Tanh())

        self.layer1.append(nn.BatchNorm1d(self.e_num[-2]))
        self.layer1.append(nn.Linear(self.e_num[-2], self.e_num[-1]))





        self.layer2 = nn.Sequential(nn.BatchNorm1d(self.e_num[-1]),nn.Linear(self.e_num[-1], self.e_num[-1]),nn.Tanh())
        self.layer3 = nn.Sequential(nn.BatchNorm1d(self.e_num[-1]),nn.Linear(self.e_num[-1], self.e_num[-1]),nn.Tanh())
        self.Tanh=nn.Tanh()





    def forward(self, x):
        x=self.layer1(x)


        mean = self.layer2(x)

        var = self.layer3(x)

        x=torch.cat([mean,var],1)





        return x


            # add position embedding


class Decoder (nn.Module):
    def __init__(self, dim, e_num, euc_num, d_num):
        super().__init__()
        self.e_num = e_num
        self.euc_num = euc_num
        self.d_num = d_num
        self.layer1 = nn.Sequential(nn.BatchNorm1d(self.e_num[-1]),nn.Linear(e_num[-1],d_num[0]),nn.Tanh())
        for i in range(len(self.d_num)-1):
            self.layer1.append(nn.BatchNorm1d(d_num[i]))

            self.layer1.append(nn.Linear(d_num[i], d_num[i+1]))
            self.layer1.append(nn.Tanh())

        self.layer1.append(nn.Linear(d_num[-1], dim))





        self.dim = dim












    def forward(self, x):

        x = self.layer1(x)
        return x



class CD_VAE_1(nn.Module):
    def __init__(self, dim,e_num,euc_num,d_num):
        super().__init__()





        self.dim = dim




        self.c_encoder =C_Encoder(dim,e_num,euc_num,d_num)

        self.decoder =Decoder(dim,e_num,euc_num,d_num)





    def forward(self, x, label=0):

        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]

        if label==0:


            # add position embedding

            casual = self.c_encoder(x)
            # classification: using cls_token output
            mean_c =casual[:,:casual.shape[-1]//2]
            var_c = casual[:,casual.shape[-1]//2:]

            # classification: using cls_token output

            z_c = mean_c + torch.randn_like(mean_c) * var_c


            rec_x = self.decoder(z_c)
            return mean_c,var_c,z_c, rec_x
        if label==1:
            casual = self.c_encoder(x)
            # classification: using cls_token output
            mean_c = casual[:, :casual.shape[-1] // 2]
            var_c = casual[:, casual.shape[-1] // 2:]
            z_c = mean_c + torch.randn_like(mean_c) * var_c

            return mean_c, var_c,  z_c
        if label == 2:

            rec_x = self.decoder(x)
            return rec_x
class RLV1(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim

        self.score_layer = MVnet(dim)
        self.norm1 = nn.LayerNorm(dim)

        self.score_token = nn.Parameter(torch.randn(1, 1, dim))
        self.score_map=nn.Sequential(nn.Linear(dim,1),nn.Sigmoid())


    def forward(self, x):
        x = torch.cat([repeat(self.score_token, '() n d -> b n d', b=x.shape[0]), x[:, :self.dim :].unsqueeze(1)], 1)


        x = (self.score_layer(self.norm1(x)))
        x=self.score_map(x[:, 0, :])


        return x
import torch.nn.functional as F
class MVnet(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        # dim = dim // 4
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.sr_ratio = sr_ratio
        # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.norm = nn.LayerNorm(dim)


    def forward(self, x):


        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_ = self.norm(x)
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        return x


class RLC1(nn.Module):
    def __init__(self, dim,num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim

        self.mean_val_layer=MVnet(dim)
        self.norm1 = nn.LayerNorm(dim)


        self.mean_token = nn.Parameter(torch.randn(1, 1, dim))
        self.var_token = nn.Parameter(torch.randn(1, 1, dim))


    def forward(self, x):
        x = torch.cat([repeat(self.mean_token, '() n d -> b n d', b=x.shape[0]),
                       repeat(self.var_token, '() n d -> b n d', b=x.shape[0]), x[:, :self.dim].unsqueeze(1)], 1)

        x = (self.mean_val_layer(self.norm1(x)))




        return torch.cat([x[:,0,:],x[:,1,:]],1)