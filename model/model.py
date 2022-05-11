import torch
import torch.nn as nn

from .components import ConvLayer2D, SliceAttention, ASCMM, CAAM, AttributePrediction

class Network(nn.Module):

    def __init__(self, opts, n_features):
        super(Network, self).__init__()
        self.initial_depth = opts.initial_depth if hasattr(opts, 'initial_depth') else 32

        channels = [self.initial_depth, self.initial_depth*2, self.initial_depth*4, self.initial_depth*8]
        dims = [64, int(64//2), int(64//4), int(64//8)]
    
        self.conv_1_1 = ConvLayer2D(1, channels[0], 3, 1, 1)
        self.conv_1_2 = ConvLayer2D(channels[0], channels[0], 3, 1, 1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2_1 = ConvLayer2D(channels[0], channels[1], 3, 1, 1)
        self.conv_2_2 = ConvLayer2D(channels[1], channels[1], 3, 1, 1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3_1 = ConvLayer2D(channels[1], channels[2], 3, 1, 1)
        self.conv_3_2 = ConvLayer2D(channels[2], channels[2], 3, 1, 1)
        self.conv_3_3 = ConvLayer2D(channels[2], channels[2], 3, 1, 1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_4_1 = ConvLayer2D(channels[2], channels[3], 3, 1, 1)
        self.conv_4_2 = ConvLayer2D(channels[3], channels[3], 3, 1, 1)
        self.conv_4_3 = ConvLayer2D(channels[3], channels[3], 3, 1, 1)

        self.gap = nn.AvgPool2d(dims[-1])

        self.slice_attn = SliceAttention(channels[3])

        self.ascmm = ASCMM(channels[3], n_features)

        self.caam = CAAM(channels[3], n_features)

        self.attr_pred = AttributePrediction(channels[3], n_features)


    def forward(self, in_img):

        conv_1_1 = self.conv_1_1(in_img)
        conv_1_2 = self.conv_1_2(conv_1_1)
        pool_1 = self.pool_1(conv_1_2)

        conv_2_1 = self.conv_2_1(pool_1)
        conv_2_2 = self.conv_2_2(conv_2_1)
        pool_2 = self.pool_2(conv_2_2)

        conv_3_1 = self.conv_3_1(pool_2)
        conv_3_2 = self.conv_3_2(conv_3_1)
        conv_3_3 = self.conv_3_3(conv_3_2)
        pool_3 = self.pool_3(conv_3_3)

        conv_4_1 = self.conv_4_1(pool_3)
        conv_4_2 = self.conv_4_2(conv_4_1)
        conv_4_3 = self.conv_4_3(conv_4_2)

        conv_4_3_gap = torch.squeeze(self.gap(conv_4_3), dim=-1)
        conv_4_3_gap = torch.squeeze(conv_4_3_gap, dim=-1)
        
        slice_attn_fv, slice_weights = self.slice_attn(conv_4_3_gap)

        feature_vectors, feature_preds_aux = self.ascmm(slice_attn_fv)

        feature_vectors_weighted, caam_weights = self.caam(feature_vectors)
        
        feature_preds = self.attr_pred(feature_vectors_weighted)

        final = torch.sigmoid(feature_preds)
        aux = torch.sigmoid(feature_preds_aux)

        return final, aux, slice_weights, caam_weights