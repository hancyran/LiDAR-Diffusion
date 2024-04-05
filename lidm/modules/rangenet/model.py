#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Backbone(nn.Module):
    """
       Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, params):
        super(Backbone, self).__init__()
        self.use_range = params["input_depth"]["range"]
        self.use_xyz = params["input_depth"]["xyz"]
        self.use_remission = params["input_depth"]["remission"]
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]
        self.OS = params["OS"]
        self.layers = params["extra"]["layers"]

        # input depth calc
        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)

        # stride play
        self.strides = [2, 2, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s

        # make the new stride
        if self.OS > current_os:
            print("Can't do OS, ", self.OS,
                  " because it is bigger than original ", current_os)
        else:
            # redo strides according to needed stride
            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break

        # check that darknet exists
        assert self.layers in model_blocks.keys()

        # generate layers depending on darknet type
        self.blocks = model_blocks[self.layers]

        # input layer
        self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        # encoder
        self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0],
                                         stride=self.strides[0], bn_d=self.bn_d)
        self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                         stride=self.strides[1], bn_d=self.bn_d)
        self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                         stride=self.strides[2], bn_d=self.bn_d)
        self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                         stride=self.strides[3], bn_d=self.bn_d)
        self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                         stride=self.strides[4], bn_d=self.bn_d)

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 1024

    # make layer useful function
    def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1):
        layers = []

        #  downsample
        layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                         kernel_size=3,
                                         stride=[1, stride], dilation=1,
                                         padding=1, bias=False)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i),
                           block(inplanes, planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def forward(self, x, return_logits=False, return_list=None):
        # filter input
        x = x[:, self.input_idxs]

        # run cnn
        # store for skip connections
        skips = {}
        out_dict = {}
        os = 1

        # first layer
        x, skips, os = self.run_layer(x, self.conv1, skips, os)
        x, skips, os = self.run_layer(x, self.bn1, skips, os)
        x, skips, os = self.run_layer(x, self.relu1, skips, os)
        if return_list and 'enc_0' in return_list:
            out_dict['enc_0'] = x.detach().cpu()  # 32, 64, 1024

        # all encoder blocks with intermediate dropouts
        x, skips, os = self.run_layer(x, self.enc1, skips, os)
        if return_list and 'enc_1' in return_list:
            out_dict['enc_1'] = x.detach().cpu()  # 64, 64, 512
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        x, skips, os = self.run_layer(x, self.enc2, skips, os)
        if return_list and 'enc_2' in return_list:
            out_dict['enc_2'] = x.detach().cpu()  # 128, 64, 256
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        x, skips, os = self.run_layer(x, self.enc3, skips, os)
        if return_list and 'enc_3' in return_list:
            out_dict['enc_3'] = x.detach().cpu()  # 256, 64, 128
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        x, skips, os = self.run_layer(x, self.enc4, skips, os)
        if return_list and 'enc_4' in return_list:
            out_dict['enc_4'] = x.detach().cpu()  # 512, 64, 64
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        x, skips, os = self.run_layer(x, self.enc5, skips, os)
        if return_list and 'enc_5' in return_list:
            out_dict['enc_5'] = x.detach().cpu()  # 1024, 64, 32
        if return_logits:
            return x

        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        if return_list is not None:
            return x, skips, out_dict
        return x, skips

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth


class Decoder(nn.Module):
    """
       Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, params, OS=32, feature_depth=1024):
        super(Decoder, self).__init__()
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]
        self.index = 0

        # stride play
        self.strides = [2, 2, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        # redo strides according to needed stride
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break

        # decoder
        self.dec5 = self._make_dec_layer(BasicBlock,
                                         [self.backbone_feature_depth, 512],
                                         bn_d=self.bn_d,
                                         stride=self.strides[0])
        self.dec4 = self._make_dec_layer(BasicBlock, [512, 256], bn_d=self.bn_d,
                                         stride=self.strides[1])
        self.dec3 = self._make_dec_layer(BasicBlock, [256, 128], bn_d=self.bn_d,
                                         stride=self.strides[2])
        self.dec2 = self._make_dec_layer(BasicBlock, [128, 64], bn_d=self.bn_d,
                                         stride=self.strides[3])
        self.dec1 = self._make_dec_layer(BasicBlock, [64, 32], bn_d=self.bn_d,
                                         stride=self.strides[4])

        # layer list to execute with skips
        self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 32

    def _make_dec_layer(self, block, planes, bn_d=0.1, stride=2):
        layers = []

        #  downsample
        if stride == 2:
            layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                                                        kernel_size=[1, 4], stride=[1, 2],
                                                        padding=[0, 1])))
        else:
            layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                             kernel_size=3, padding=1)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        layers.append(("residual", block(planes[1], planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)  # up
        if feats.shape[-1] > x.shape[-1]:
            os //= 2  # match skip
            feats = feats + skips[os].detach()  # add skip
        x = feats
        return x, skips, os

    def forward(self, x, skips, return_logits=False, return_list=None):
        os = self.backbone_OS
        out_dict = {}

        # run layers
        x, skips, os = self.run_layer(x, self.dec5, skips, os)
        if return_list and 'dec_4' in return_list:
            out_dict['dec_4'] = x.detach().cpu()  # 512, 64, 64
        x, skips, os = self.run_layer(x, self.dec4, skips, os)
        if return_list and 'dec_3' in return_list:
            out_dict['dec_3'] = x.detach().cpu()  # 256, 64, 128
        x, skips, os = self.run_layer(x, self.dec3, skips, os)
        if return_list and 'dec_2' in return_list:
            out_dict['dec_2'] = x.detach().cpu()  # 128, 64, 256
        x, skips, os = self.run_layer(x, self.dec2, skips, os)
        if return_list and 'dec_1' in return_list:
            out_dict['dec_1'] = x.detach().cpu()  # 64, 64, 512
        x, skips, os = self.run_layer(x, self.dec1, skips, os)
        if return_list and 'dec_0' in return_list:
            out_dict['dec_0'] = x.detach().cpu()  # 32, 64, 1024

        logits = torch.clone(x).detach()
        x = self.dropout(x)

        if return_logits:
            return x, logits
        if return_list is not None:
            return out_dict
        return x

    def get_last_depth(self):
        return self.last_channels


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = Backbone(params=self.config["backbone"])
        self.decoder = Decoder(params=self.config["decoder"], OS=self.config["backbone"]["OS"],
                               feature_depth=self.backbone.get_last_depth())

    def load_pretrained_weights(self, path):
        w_dict = torch.load(path + "/backbone",
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        w_dict = torch.load(path + "/segmentation_decoder",
                            map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)

    def forward(self, x, return_logits=False, return_final_logits=False, return_list=None, agg_type='depth'):
        if return_logits:
            logits = self.backbone(x, return_logits)
            logits = F.adaptive_avg_pool2d(logits, (1, 1)).squeeze()
            logits = torch.clone(logits).detach().cpu().numpy()
            return logits
        elif return_list is not None:
            x, skips, enc_dict = self.backbone(x, return_list=return_list)
            dec_dict = self.decoder(x, skips, return_list=return_list)
            out_dict = {**enc_dict, **dec_dict}
            return out_dict
        elif return_final_logits:
            assert agg_type in ['all', 'sector', 'depth']
            y, skips = self.backbone(x)
            y, logits = self.decoder(y, skips, True)

            B, C, H, W = logits.shape
            N = 16

            # avg all
            if agg_type == 'all':
                logits = logits.mean([2, 3])
            # avg in patch
            elif agg_type == 'sector':
                logits = logits.view(B, C, H, N, W // N).mean([2, 4]).reshape(B, -1)
            # avg in row
            elif agg_type == 'depth':
                logits = logits.view(B, C, N, H // N, W).mean([3, 4]).reshape(B, -1)

            logits = torch.clone(logits).detach().cpu().numpy()
            return logits
        else:
            y, skips = self.backbone(x)
            y = self.decoder(y, skips, False)
            return y
