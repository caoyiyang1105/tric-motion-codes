import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat
from pytorch_wavelets import DWT2D, IDWT2D

from model.networks.ccmd import CausalModelingBlk, CCMD, TDEFusionModule
from model.networks.gcn import GCN, get_humanml3d_adjacency_matrix, normalize_adjacency_matrix

class TriCModel(nn.Module):
    def __init__(self, d_model, nlayers, nhead, dim_feedforward=512, dropout=0.1, is_training=False, ds_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList([
            TriCBlk(d_model, nhead, dim_feedforward, dropout,is_training=is_training, ds_ratio=ds_ratio)
            for _ in range(nlayers)
        ])
        self.is_training = is_training

    def forward(self, x, frames_mask=None, sentence_embed=None, word_embed=None, text_mask=None):
        casual_lst = []
        for layer in self.layers:
            x, x_casual = layer(x, frames_mask=frames_mask,
                      sentence_embed=sentence_embed, word_embed=word_embed, text_mask=text_mask)
            casual_lst.append(x_casual)
        # x = casual_lst[-1][2]
        return x, casual_lst

class TriCBlk(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, is_training=False, ds_ratio=4):
        
        super().__init__()
        self.is_training=is_training
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # domain modeling
        self.spa_modeling_blk = SpatialModelingBlk(d_model, dropout=dropout)
        self.temp_modeling_blk = TemporalModelingBlk(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.freq_modeling_blk = FreqModelingBlk(d_model, num_groups=16)
        
        # domain fusion
        self.fusion = ScoreFusionBlock(d_model)

        # text injection
        self.cond_injector = TextCondInjector(
            d_model, nhead, dropout=dropout)

        # causal modeling
        self.spa_causal_blk = CausalModelingBlk(d_model)
        self.temp_causal_blk = CausalModelingBlk(d_model)
        self.freq_causal_blk = CausalModelingBlk(d_model)

        # causal Fusion
        self.tde_fusion = TDEFusionModule(256, 256)  # 3 domains
        self.out = nn.Linear(d_model, ds_ratio*12)

    def forward(self, x, frames_mask=None, sentence_embed=None, word_embed=None, text_mask=None):
        # x: [bs, 196, 22, 256]
        # frames_mask: [bs, 196]
        # time_emb: [bs, 256]
        # word_embed: [bs, word_len, 256]
        # text_mask: [bs, word_len]

        x_orig = x

        # spatial modeling
        x_spa = self.spa_modeling_blk(x)
        # temporal modeling
        x_temp = self.temp_modeling_blk(x, frames_mask=frames_mask)
        # frequency modeling
        x_freq = self.freq_modeling_blk(x)

        # fusion
        x = self.fusion(x_orig, x_spa, x_temp, x_freq, sentence_embed)

        # text conditioning
        x = self.cond_injector(x, word_embed=word_embed, text_mask=text_mask)

        # causal modeling
        tde, spa_tde, temp_tde, freq_tde = None, None, None, None
        if self.is_training:
            spa_tde = self.spa_causal_blk(x_spa[:, 1:])
            temp_tde = self.temp_causal_blk(x_temp[:, 1:])
            freq_tde = self.freq_causal_blk(x_freq[:, 1:])
            fused_tde = self.tde_fusion(spa_tde, temp_tde, freq_tde)
            tde = self.out(fused_tde)

        return x, tde

class ScoreFusionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.local_gate = nn.Sequential( 
            nn.Linear(3*d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, 3,  bias=False) 
        )

        self.fusion_gate = nn.Linear(d_model*2, d_model)

        self.semantic_fc = nn.Linear(d_model, d_model)

        self.spa_score = nn.Sequential(
            nn.Linear(2*d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, 1, bias=False)
        )
        self.temp_score = nn.Sequential(
            nn.Linear(2*d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, 1, bias=False)
        )
        self.freq_score = nn.Sequential(
            nn.Linear(2*d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, 1, bias=False)
        )

    def forward(self, x_orig, x_spa, x_temp, x_freq, sentence_embed):
        b, t, j, d = x_orig.shape

        x_cat = torch.cat([x_spa, x_temp, x_freq], dim=-1)    # [B,T,J,3D]
        local_logits = self.local_gate(x_cat)                     # [B,T,J,3]

        sentence_embed = self.semantic_fc(sentence_embed).squeeze(0)
        # if sentence_embed.shape[0] !=1:
        #     sentence_embed = sentence_embed.squeeze(0)
        sentence_embed = sentence_embed.unsqueeze(
            1).unsqueeze(2).expand(-1, t, j, -1)

        spa_score = self.spa_score(torch.cat([sentence_embed, x_spa], dim=-1))
        temp_score = self.temp_score(
            torch.cat([sentence_embed, x_temp], dim=-1))
        freq_score = self.freq_score(
            torch.cat([sentence_embed, x_freq], dim=-1))

        global_logits = torch.cat([spa_score, temp_score, freq_score], dim=-1)

        logits = local_logits + global_logits               # Broadcast
        alpha = torch.softmax(logits, dim=-1)              # [B,T,J,3]
        a_s, b_t, c_f = alpha[..., 0:1], alpha[..., 1:2], alpha[..., 2:3]

        x_fused = a_s * x_spa + b_t * x_temp + c_f * x_freq     # [B,T,J,D]

        gate = self.fusion_gate(torch.cat([x_orig, x_fused], dim=-1)).sigmoid()
        x = x_orig * (1 - gate) + x_fused * gate
        return x


class TemporalModelingBlk(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.temp_trans = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True)

    def forward(self, x, frames_mask=None):
        B, T, J, D = x.shape
        x = rearrange(x, 'b t j d -> (b j) t d')  # (B*J, T, D)
        frames_mask = repeat(frames_mask, 'b t -> (b j) t', j=J)
        x = self.temp_trans(x, src_key_padding_mask=frames_mask)
        x = rearrange(x, '(b j) t d -> b t j d', b=B, j=J)  # (B, T, J, D)
        return x


class SpatialModelingBlk(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout

        joints_adj_fc = get_humanml3d_adjacency_matrix()
        normalized_joints_adj = normalize_adjacency_matrix(joints_adj_fc)
        self.adj = torch.FloatTensor(normalized_joints_adj)

        self.spa_modeling_layer = GCN(d_model, d_model, d_model, dropout)

    def forward(self, x):
        b, t, j, d = x.shape
        x = rearrange(x, 'b t j d -> (b t) j d')
        self.adj = self.adj.to(x.device)
        x = self.spa_modeling_layer(x, self.adj)
        x = rearrange(x, '(b t) j d -> b t j d', b=b, j=j)
        return x


class TextCondInjector(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead


        self.norm = nn.LayerNorm(d_model)
        self.cond_injector_layer = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)


    def forward(self, x, word_embed=None, text_mask=None):
        b, t, j, d = x.shape
        b, word_len, d = word_embed.shape

        x1 = x[:, 1:]
        x1 = rearrange(x1, 'b t j d -> (b j) t d')

        identity = x1

        word_embed = repeat(word_embed, 'b t d -> (b j) t d', j=j)
        text_mask = repeat(text_mask, 'b t -> (b j) t', j=j)

        x1 = self.norm(x1)
        x1, _ = self.cond_injector_layer(
            x1, word_embed, word_embed, key_padding_mask=text_mask)

        # residual
        x1 += identity

        x1 = rearrange(x1, '(b j) t d -> b t j d', b=b, j=j)
        out = torch.cat([x[:, :1], x1], dim=1)

        return out

class FreqModelingBlk(nn.Module):
    def __init__(self, d_model, num_groups=16):
        super().__init__()

        self.d_model = d_model

        self.dwt = DWT2D(wave='haar', mode='reflect')
        self.idwt = IDWT2D(wave='haar', mode='reflect')


        d_model_l = d_model * 2
        self.low_norm = nn.LayerNorm(d_model_l)
        self.low_act = nn.GELU()
        self.se = JointTemporalSE(d_model_l, reduction=8)

        d_model_h = d_model * 3
        self.temporal_conv = nn.Conv2d(
            in_channels=d_model_h,
            out_channels=d_model_h,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=d_model_h
        )

        self.joint_conv = nn.Conv2d(
            in_channels=d_model_h,
            out_channels=d_model_h,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=d_model_h
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels=d_model_h,
            out_channels=d_model_h,
            kernel_size=1
        )

        self.high_norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=d_model_h)
        self.high_act = nn.GELU()

    def forward(self, x):
        # ll: [bs, 98, 11, 256]
        b, t, j, d = x.shape

        identity_x = x

        x1 = x[:, 1:]
        x1 = rearrange(x1, 'b t j d -> b d t j')
        ll, h = self.dwt(x1)

        h = h[0]
        lh = h[:, :, 0, :, :]
        hl = h[:, :, 1, :, :]
        hh = h[:, :, 2, :, :]

        ll_fft = torch.fft.fft2(ll)
        ll_magnitude = torch.abs(ll_fft)
        ll_phase = torch.angle(ll_fft)

        ll = torch.cat([ll_magnitude, ll_phase], dim=1)
        identity = ll

        ll = ll.permute(0, 2, 3, 1)
        ll = self.low_norm(ll)
        ll = ll.permute(0, 3, 1, 2)
        ll = self.se(ll)
        ll = self.low_act(ll)
        ll += identity
        ll_magnitude, ll_phase = torch.chunk(ll, 2, dim=1)

        h3 = torch.cat([lh, hl, hh], dim=1)  # [b, 3*d, t, j]
        identity = h3
        h3_temporal = self.temporal_conv(h3)
        h3_joint = self.joint_conv(h3)
        h3 = h3_temporal + h3_joint

        h3 = self.pointwise_conv(h3)
        h3 = self.high_norm(h3)
        h3 = self.high_act(h3)
        h3 += identity

        lh, hl, hh = torch.chunk(h3, 3, dim=1)

        ll_fft = torch.polar(ll_magnitude, ll_phase)

        ll = torch.fft.ifft2(ll_fft).real

        h = torch.stack([lh, hl, hh], dim=2)
        h = [h]
        x2 = (ll, h)

        x2 = self.idwt(x2)

        x2 = rearrange(x2, 'b d t j -> b t j d')

        x = torch.cat([x[:, :1], x2], dim=1)

        # residual
        if self.ds_ratio ==7:
            x = x + identity_x
        elif self.ds_ratio ==4:
            x = x[:,:-1] + identity_x
        return x


class JointTemporalSE(nn.Module):
    def __init__(self, dim, reduction=8):
        super(JointTemporalSE, self).__init__()

        self.temporal_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)), 
            nn.Conv1d(dim, dim // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.joint_se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)), 
            nn.Conv1d(dim, dim // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        temp_y = self.temporal_se[0](x)
        temp_y = temp_y.squeeze(-1)
        temp_y = self.temporal_se[1:](temp_y)
        temp_y = temp_y.unsqueeze(-1)

        joint_y = self.joint_se[0](x)
        joint_y = joint_y.squeeze(-2)
        joint_y = self.joint_se[1:](joint_y)

        joint_y = joint_y.unsqueeze(2)

        y = temp_y * joint_y 
        return x * y 
