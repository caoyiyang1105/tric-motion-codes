from torch import nn
import torch

class CausalModelingBlk(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.causal_blk_f = CCMD(d_model)
        self.causal_blk_cf = CCMD(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        w_f = self.causal_blk_f(x)
        w_cf = self.causal_blk_cf(x)

        x_f = x * w_f
        x_cf = x * w_cf

        x_f = self.proj(x_f)
        x_cf = self.proj(x_cf)

        tde = x_f - x_cf
        
        return tde

class CCMD(nn.Module):
    def __init__(self, feat_dim, reduction = 8):
        super().__init__()
        self.pool_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_max = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // reduction, feat_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = x.permute(0, 3, 1, 2)
        z1 = self.pool_avg(z)
        z2 = self.pool_max(z)
        z = z1+z2
        # squeeze -> [B, C]
        z = z.view(z.size(0), -1)
        # FC -> [B, C]
        w = self.fc(z)
        # reshape -> [B, 1, 1, C]
        w = w.view(w.size(0), 1, 1, w.size(1))
        return w


class TDEFusionModule(nn.Module):
    def __init__(self, input_dim, fusion_dim=256):
        super(TDEFusionModule, self).__init__()
        
        self.weights = nn.Parameter(torch.ones(3) / 3)
        self.feature_transform = nn.Linear(input_dim, fusion_dim)

    def forward(self, x_spa, x_temp, x_freq):
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        fused_feat = (normalized_weights[0] * x_spa + 
                     normalized_weights[1] * x_temp + 
                     normalized_weights[2] * x_freq)

        output = self.feature_transform(fused_feat)
        
        return output
