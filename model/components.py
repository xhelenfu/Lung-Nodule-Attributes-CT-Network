import torch
import torch.nn as nn

from .intialisation import init_weights

class ConvLayer2D(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding):
        super(ConvLayer2D, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=k_size,
                      padding=padding, stride=stride, bias=True),
            nn.ReLU(inplace=True)
        )
                                           
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        return self.conv_unit(inputs)


class SliceAttention(nn.Module):
    def __init__(self, in_ch):
        super(SliceAttention, self).__init__()
        self.slice_mlp = nn.Sequential(
            nn.Linear(in_ch, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True),
            nn.Softmax(dim=0)
        )
        
        for m in self.children():
            init_weights(m, init_type='glorot')

    def forward(self, inputs):
        attn_weights = self.slice_mlp(inputs)
        weighted = attn_weights.repeat(1,inputs.shape[1]) * inputs
        weighted_sum = torch.sum(weighted, dim=0)
        return weighted_sum.unsqueeze(0), torch.transpose(attn_weights,0,1)


class ASCMM(nn.Module):
    def __init__(self, in_ch, n_features):
        super(ASCMM, self).__init__()
        self.n_features = n_features

        for i in range(1, n_features+1):
            feature_mlp = nn.Sequential(
                nn.Linear(in_ch, in_ch, bias=True),
                nn.Linear(in_ch, 1, bias=False)
            )
            setattr(self, 'fv_mlp_%d' %i, feature_mlp)

        for m in self.children():
            init_weights(m, init_type='glorot')

    def forward(self, inputs):
        pred_all = None
        fv_all = None

        for i in range(1, self.n_features+1):
            mlp = getattr(self, 'fv_mlp_%d' %i)
            pred = mlp(inputs)
            fv = mlp[0](inputs)

            if pred_all is None:
                pred_all = pred[:]
                fv_all = fv[:]
            else:
                pred_all = torch.cat((pred_all, pred), 1)
                fv_all = torch.cat((fv_all, fv), 0)
        
        return fv_all, pred_all
        

class CAAM(nn.Module):
    def __init__(self, in_ch, n_features):
        super(CAAM, self).__init__()
        self.n_features = n_features

        for i in range(1, n_features+1):
            feature_mlp = nn.Sequential(
                nn.Linear(in_ch, 128, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1, bias=True)
            )
            setattr(self, 'fweight_mlp_%d' %i, feature_mlp)

        for m in self.children():
            init_weights(m, init_type='glorot')

    def forward(self, inputs):
        weighted_all = None
        weights_all = None

        for i in range(1, self.n_features+1):
            mlp = getattr(self, 'fweight_mlp_%d' %i)
            attn_weights = mlp(inputs)

            weighted = attn_weights.repeat(1,inputs.shape[1]) * inputs
            weighted_sum = torch.sum(weighted, dim=0)

            if weighted_all is None:
                weighted_all = weighted_sum.unsqueeze(0)
                weights_all = torch.transpose(attn_weights,0,1)
            else:
                weighted_all = torch.cat((weighted_all, weighted_sum.unsqueeze(0)), 0)
                weights_all = torch.cat((weights_all, torch.transpose(attn_weights,0,1)), 0)
        
        return weighted_all, weights_all


class AttributePrediction(nn.Module):
    def __init__(self, in_ch, n_features):
        super(AttributePrediction, self).__init__()
        self.n_features = n_features

        for i in range(1, n_features+1):
            feature_mlp = nn.Sequential(
                nn.Linear(in_ch, in_ch, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_ch, 1, bias=False)
            )
            setattr(self, 'feature_mlp_%d' %i, feature_mlp)

        for m in self.children():
            init_weights(m, init_type='glorot')

    def forward(self, inputs):
        pred_all = None

        for i in range(1, self.n_features+1):
            mlp = getattr(self, 'feature_mlp_%d' %i)
            pred = mlp(inputs[i-1,:].unsqueeze(0))

            if pred_all is None:
                pred_all = pred[:]
            else:
                pred_all = torch.cat((pred_all, pred), 1)
        
        return pred_all