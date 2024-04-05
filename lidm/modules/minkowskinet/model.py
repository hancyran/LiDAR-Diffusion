import torch
import torch.nn as nn

try:
    import torchsparse
    import torchsparse.nn as spnn
    from ..ts import basic_blocks
except ImportError:
    raise Exception('Required ts lib. Reference: https://github.com/mit-han-lab/torchsparse/tree/v1.4.0')


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        cr = config.model_params.cr
        cs = config.model_params.layer_num
        cs = [int(cr * x) for x in cs]

        self.pres = self.vres = config.model_params.voxel_size
        self.num_classes = config.model_params.num_class

        self.stem = nn.Sequential(
            spnn.Conv3d(config.model_params.input_dims, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            basic_blocks.BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            basic_blocks.ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            basic_blocks.ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            basic_blocks.BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            basic_blocks.ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            basic_blocks.ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            basic_blocks.BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            basic_blocks.ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            basic_blocks.ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            basic_blocks.BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            basic_blocks.ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            basic_blocks.ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            basic_blocks.BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                basic_blocks.ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1,
                                           dilation=1),
                basic_blocks.ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            basic_blocks.BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                basic_blocks.ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1,
                                           dilation=1),
                basic_blocks.ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            basic_blocks.BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                basic_blocks.ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1,
                                           dilation=1),
                basic_blocks.ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            basic_blocks.BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                basic_blocks.ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1,
                                           dilation=1),
                basic_blocks.ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], self.num_classes))

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data_dict, return_logits=False, return_final_logits=False):
        x = data_dict['lidar']
        x.C = x.C.int()

        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        if return_logits:
            output_dict = dict()
            output_dict['logits'] = x4.F
            output_dict['batch_indices'] = x4.C[:, -1]
            return output_dict

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        if return_final_logits:
            output_dict = dict()
            output_dict['logits'] = y4.F
            output_dict['coords'] = y4.C[:, :3]
            output_dict['batch_indices'] = y4.C[:, -1]
            return output_dict

        output = self.classifier(y4.F)
        data_dict['output'] = output.F

        return data_dict
