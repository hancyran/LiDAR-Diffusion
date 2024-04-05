import hashlib
import os

import requests
import torch
import torch.nn as nn

from tqdm import tqdm

from . import l1, l2
from ...utils.model_utils import build_model

URL_MAP = {
}

CKPT_MAP = {
}

MD5_MAP = {
}

PERCEPTUAL_TYPE = {
    'rangenet_full': [('enc_0', 32), ('enc_1', 64), ('enc_2', 128), ('enc_3', 256), ('enc_4', 512), ('enc_5', 1024),
                      ('dec_4', 512), ('dec_3', 256), ('dec_2', 128), ('dec_1', 64), ('dec_0', 32)],
    'rangenet_enc': [('enc_0', 32), ('enc_1', 64), ('enc_2', 128), ('enc_3', 256), ('enc_4', 512), ('enc_5', 1024)],
    'rangenet_dec': [('dec_4', 512), ('dec_3', 256), ('dec_2', 128), ('dec_1', 64), ('dec_0', 32)],
    'rangenet_final': [('dec_0', 32)]
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class PerceptualLoss(nn.Module):
    def __init__(self, ptype, depth_scale, log_scale=True, use_dropout=True, lpips=False, p_loss='l1'):
        super().__init__()
        self.depth_scale = depth_scale
        self.log_scale = log_scale

        if p_loss == "l1":
            self.p_loss = l1
        else:
            self.p_loss = l2

        self.chns = PERCEPTUAL_TYPE[ptype]
        self.return_list = [name for name, _ in self.chns]
        self.loss_scale = [5.0, 3.39, 2.29, 1.61, 0.895]  # predefined based on the loss of each stage after a few epochs (refer )
        self.net = build_model('kitti', 'rangenet')
        self.lin_list = nn.ModuleList([NetLinLayer(ch, use_dropout=use_dropout) for _, ch in self.chns]) if lpips else None
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def normalize_tensor(x, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    @staticmethod
    def spatial_average(x, keepdim=True):
        return x.mean([2, 3], keepdim=keepdim)

    def preprocess(self, *inputs):
        assert len(inputs) == 2, 'input with both depth images and coord images'
        depth_img, xyz_img = inputs

        # scale to standard rangenet input
        depth_img = (depth_img * 0.5 + 0.5) * self.depth_scale
        if self.log_scale:
            depth_img = torch.exp2(depth_img) - 1

        img = torch.cat([depth_img, xyz_img], 1)
        return img

    def forward(self, target, input):
        in0_input, in1_input = self.preprocess(*input), self.preprocess(*target)
        outs0, outs1 = self.net(in0_input, return_list=self.return_list), self.net(in1_input, return_list=self.return_list)

        val_list = []
        for i, (name, _) in enumerate(self.chns):
            feats0, feats1 = self.normalize_tensor(outs0[name].to(in0_input.device)), \
                             self.normalize_tensor(outs1[name].to(in0_input.device))
            diffs = self.p_loss(feats0, feats1)
            res = self.lin_list[i].model(diffs) if self.lin_list is not None else diffs.mean(1, keepdim=True)
            res = self.spatial_average(res, keepdim=True) * self.loss_scale[i]
            val_list.append(res)
        val = sum(val_list)
        return val
