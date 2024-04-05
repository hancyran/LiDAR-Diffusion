"""
@Author: Haoxi Ran
@Date: 01/03/2024
@Citation: Towards Realistic Scene Generation with LiDAR Diffusion Models

"""

import math
from itertools import repeat
from typing import List, Tuple, Union
import numpy as np
import torch

from . import build_model, VOXEL_SIZE, MODALITY2MODEL, MODAL2BATCHSIZE, DATASET_CONFIG, AGG_TYPE, NUM_SECTORS, \
    TYPE2DATASET, DATA_CONFIG

try:
    from torchsparse import SparseTensor, PointTensor
    from torchsparse.utils.collate import sparse_collate_fn
    from .modules.chamfer3D.dist_chamfer_3D import chamfer_3DDist
    from .modules.chamfer2D.dist_chamfer_2D import chamfer_2DDist
    from .modules.emd.emd_module import emdModule
except:
    print(
        'To install torchsparse 1.4.0, please refer to https://github.com/mit-han-lab/torchsparse/tree/74099d10a51c71c14318bce63d6421f698b24f24')


def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_quantize(coords, voxel_size: Union[float, Tuple[float, ...]] = 1, *, return_index: bool = False,
                    return_inverse: bool = False) -> List[np.ndarray]:
    """
    Modified based on https://github.com/mit-han-lab/torchsparse/blob/462dea4a701f87a7545afb3616bf2cf53dd404f3/torchsparse/utils/quantize.py

    """
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, coords.shape[1]))
    assert isinstance(voxel_size, tuple) and len(voxel_size) in [2, 3]  # support 2D and 3D coordinates only

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(
        ravel_hash(coords), return_index=True, return_inverse=True
    )
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs


def pcd2range(pcd, size, fov, depth_range, remission=None, labels=None, **kwargs):
    # laser parameters
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth (distance) of all points
    depth = np.linalg.norm(pcd, 2, axis=1)

    # mask points out of range
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    depth, pcd = depth[mask], pcd[mask]

    # get scan components
    scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov_range  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= size[1]  # in [0.0, W]
    proj_y *= size[0]  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]
    proj_y = np.maximum(0, np.minimum(size[0] - 1, np.floor(proj_y))).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    proj_x, proj_y = proj_x[order], proj_y[order]

    # project depth
    depth = depth[order]
    proj_range = np.full(size, -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = depth

    # project point feature
    if remission is not None:
        remission = remission[mask][order]
        proj_feature = np.full(size, -1, dtype=np.float32)
        proj_feature[proj_y, proj_x] = remission
    elif labels is not None:
        labels = labels[mask][order]
        proj_feature = np.full(size, 0, dtype=np.float32)
        proj_feature[proj_y, proj_x] = labels
    else:
        proj_feature = None

    return proj_range, proj_feature


def range2xyz(range_img, fov, depth_range, depth_scale, log_scale=True, **kwargs):
    # laser parameters
    size = range_img.shape
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    if log_scale:
        depth = (np.exp2(range_img * depth_scale) - 1)
    else:
        depth = range_img

    scan_x, scan_y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    scan_x = scan_x.astype(np.float64) / size[1]
    scan_y = scan_y.astype(np.float64) / size[0]

    yaw = np.pi * (scan_x * 2 - 1)
    pitch = (1.0 - scan_y) * fov_range - abs(fov_down)

    xyz = -np.ones((3, *size))
    xyz[0] = np.cos(yaw) * np.cos(pitch) * depth
    xyz[1] = -np.sin(yaw) * np.cos(pitch) * depth
    xyz[2] = np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    xyz[:, ~mask] = -1

    return xyz


def pcd2voxel(pcd):
    pcd_voxel = np.round(pcd / VOXEL_SIZE)
    pcd_voxel = pcd_voxel - pcd_voxel.min(0, keepdims=1)
    feat = np.concatenate((pcd, -np.ones((pcd.shape[0], 1))), axis=1)  # -1 for remission placeholder
    _, inds, inverse_map = sparse_quantize(pcd_voxel, 1, return_index=True, return_inverse=True)

    feat = torch.FloatTensor(feat[inds])
    pcd_voxel = torch.LongTensor(pcd_voxel[inds])
    lidar = SparseTensor(feat, pcd_voxel)
    output = {'lidar': lidar}
    return output


def pcd2voxel_full(data_type, *args):
    config = DATA_CONFIG[data_type]
    x_range, y_range, z_range = config['x'], config['y'], config['z']
    vol_shape = (math.ceil((x_range[1] - x_range[0]) / VOXEL_SIZE), math.ceil((y_range[1] - y_range[0]) / VOXEL_SIZE),
                 math.ceil((z_range[1] - z_range[0]) / VOXEL_SIZE))
    min_bound = (math.ceil((x_range[0]) / VOXEL_SIZE), math.ceil((y_range[0]) / VOXEL_SIZE),
                 math.ceil((z_range[0]) / VOXEL_SIZE))

    output = tuple()
    for data in args:
        volume_list = []
        for pcd in data:
            # mask out invalid points
            mask_x = np.logical_and(pcd[:, 0] > x_range[0], pcd[:, 0] < x_range[1])
            mask_y = np.logical_and(pcd[:, 1] > y_range[0], pcd[:, 1] < y_range[1])
            mask_z = np.logical_and(pcd[:, 2] > z_range[0], pcd[:, 2] < z_range[1])
            mask = mask_x & mask_y & mask_z
            pcd = pcd[mask]

            # voxelize
            pcd_voxel = np.floor(pcd / VOXEL_SIZE)
            _, indices, inverse_map = sparse_quantize(pcd_voxel, 1, return_index=True, return_inverse=True)
            pcd_voxel = pcd_voxel[indices]
            pcd_voxel = (pcd_voxel - min_bound).astype(np.int32)

            # 2D bev grid
            vol = np.zeros(vol_shape, dtype=np.float32)
            vol[pcd_voxel[:, 0], pcd_voxel[:, 1], pcd_voxel[:, 2]] = 1
            volume_list.append(vol)
        output += (volume_list,)
    return output


# def pcd2bev_full(data_type, *args, voxel_size=VOXEL_SIZE):
#     config = DATA_CONFIG[data_type]
#     x_range, y_range = config['x'], config['y']
#     vol_shape = (math.ceil((x_range[1] - x_range[0]) / voxel_size), math.ceil((y_range[1] - y_range[0]) / voxel_size))
#     min_bound = (math.ceil((x_range[0]) / voxel_size), math.ceil((y_range[0]) / voxel_size))
#
#     output = tuple()
#     for data in args:
#         volume_list = []
#         for pcd in data:
#             # mask out invalid points
#             mask_x = np.logical_and(pcd[:, 0] > x_range[0], pcd[:, 0] < x_range[1])
#             mask_y = np.logical_and(pcd[:, 1] > y_range[0], pcd[:, 1] < y_range[1])
#             mask = mask_x & mask_y
#             pcd = pcd[mask][:, :2]  # keep x,y coord
#
#             # voxelize
#             pcd_voxel = np.floor(pcd / voxel_size)
#             _, indices, inverse_map = sparse_quantize(pcd_voxel, 1, return_index=True, return_inverse=True)
#             pcd_voxel = pcd_voxel[indices]
#             pcd_voxel = (pcd_voxel - min_bound).astype(np.int32)
#
#             # 2D bev grid
#             vol = np.zeros(vol_shape, dtype=np.float32)
#             vol[pcd_voxel[:, 0], pcd_voxel[:, 1]] = 1
#             volume_list.append(vol)
#         output += (volume_list,)
#     return output


def pcd2bev_sum(data_type, *args, voxel_size=VOXEL_SIZE):
    config = DATA_CONFIG[data_type]
    x_range, y_range = config['x'], config['y']
    vol_shape = (math.ceil((x_range[1] - x_range[0]) / voxel_size), math.ceil((y_range[1] - y_range[0]) / voxel_size))
    min_bound = (math.ceil((x_range[0]) / voxel_size), math.ceil((y_range[0]) / voxel_size))

    output = tuple()
    for data in args:
        volume_sum = np.zeros(vol_shape, np.float32)
        for pcd in data:
            # mask out invalid points
            mask_x = np.logical_and(pcd[:, 0] > x_range[0], pcd[:, 0] < x_range[1])
            mask_y = np.logical_and(pcd[:, 1] > y_range[0], pcd[:, 1] < y_range[1])
            mask = mask_x & mask_y
            pcd = pcd[mask][:, :2]  # keep x,y coord

            # voxelize
            pcd_voxel = np.floor(pcd / voxel_size)
            _, indices, inverse_map = sparse_quantize(pcd_voxel, 1, return_index=True, return_inverse=True)
            pcd_voxel = pcd_voxel[indices]
            pcd_voxel = (pcd_voxel - min_bound).astype(np.int32)

            # summation
            volume_sum[pcd_voxel[:, 0], pcd_voxel[:, 1]] += 1.
        output += (volume_sum,)
    return output


def pcd2bev_bin(data_type, *args, voxel_size=0.5):
    config = DATA_CONFIG[data_type]
    x_range, y_range = config['x'], config['y']
    vol_shape = (math.ceil((x_range[1] - x_range[0]) / voxel_size), math.ceil((y_range[1] - y_range[0]) / voxel_size))
    min_bound = (math.ceil((x_range[0]) / voxel_size), math.ceil((y_range[0]) / voxel_size))

    output = tuple()
    for data in args:
        pcd_list = []
        for pcd in data:
            # mask out invalid points
            mask_x = np.logical_and(pcd[:, 0] > x_range[0], pcd[:, 0] < x_range[1])
            mask_y = np.logical_and(pcd[:, 1] > y_range[0], pcd[:, 1] < y_range[1])
            mask = mask_x & mask_y
            pcd = pcd[mask][:, :2]  # keep x,y coord

            # voxelize
            pcd_voxel = np.floor(pcd / voxel_size)
            _, indices, inverse_map = sparse_quantize(pcd_voxel, 1, return_index=True, return_inverse=True)
            pcd_voxel = pcd_voxel[indices]
            pcd_voxel = ((pcd_voxel - min_bound) / vol_shape).astype(np.float32)
            pcd_list.append(pcd_voxel)
        output += (pcd_list,)
    return output


def bev_sample(data_type, *args, voxel_size=0.5):
    config = DATA_CONFIG[data_type]
    x_range, y_range = config['x'], config['y']

    output = tuple()
    for data in args:
        pcd_list = []
        for pcd in data:
            # mask out invalid points
            mask_x = np.logical_and(pcd[:, 0] > x_range[0], pcd[:, 0] < x_range[1])
            mask_y = np.logical_and(pcd[:, 1] > y_range[0], pcd[:, 1] < y_range[1])
            mask = mask_x & mask_y
            pcd = pcd[mask][:, :2]  # keep x,y coord

            # voxelize
            pcd_voxel = np.floor(pcd / voxel_size)
            _, indices, inverse_map = sparse_quantize(pcd_voxel, 1, return_index=True, return_inverse=True)
            pcd = pcd[indices]
            pcd_list.append(pcd)
        output += (pcd_list,)
    return output


def preprocess_pcd(pcd, **kwargs):
    depth = np.linalg.norm(pcd, 2, axis=1)
    mask = np.logical_and(depth > kwargs['depth_range'][0], depth < kwargs['depth_range'][1])
    pcd = pcd[mask]
    return pcd


def preprocess_range(pcd, **kwargs):
    depth_img = pcd2range(pcd, **kwargs)[0]
    xyz_img = range2xyz(depth_img, log_scale=False, **kwargs)
    depth_img = depth_img[None]
    img = np.vstack([depth_img, xyz_img])
    return img


def batch2list(batch_dict, agg_type='depth', **kwargs):
    """
    Aggregation Type: Default 'depth', ['all', 'sector', 'depth']
    """
    output_list = []
    batch_indices = batch_dict['batch_indices']
    for b_idx in range(batch_indices.max() + 1):
        # avg all
        if agg_type == 'all':
            logits = batch_dict['logits'][batch_indices == b_idx].mean(0)

        # avg on sectors
        elif agg_type == 'sector':
            logits = batch_dict['logits'][batch_indices == b_idx]
            coords = batch_dict['coords'][batch_indices == b_idx].float()
            coords = coords - coords.mean(0)
            angle = torch.atan2(coords[:, 1], coords[:, 0])  # [-pi, pi]
            sector_range = torch.linspace(-np.pi - 1e-4, np.pi + 1e-4, NUM_SECTORS + 1)
            logits_list = []
            for i in range(NUM_SECTORS):
                sector_indices = torch.where((angle >= sector_range[i]) & (angle < sector_range[i + 1]))[0]
                sector_logits = logits[sector_indices].mean(0)
                sector_logits = torch.nan_to_num(sector_logits, 0.)
                logits_list.append(sector_logits)
            logits = torch.cat(logits_list)  # dim: 768

        # avg by depth
        elif agg_type == 'depth':
            logits = batch_dict['logits'][batch_indices == b_idx]
            coords = batch_dict['coords'][batch_indices == b_idx].float()
            coords = coords - coords.mean(0)
            bev_depth = torch.norm(coords, dim=-1) * VOXEL_SIZE
            sector_range = torch.linspace(kwargs['depth_range'][0] + 3, kwargs['depth_range'][1], NUM_SECTORS + 1)
            sector_range[0] = 0.
            logits_list = []
            for i in range(NUM_SECTORS):
                sector_indices = torch.where((bev_depth >= sector_range[i]) & (bev_depth < sector_range[i + 1]))[0]
                sector_logits = logits[sector_indices].mean(0)
                sector_logits = torch.nan_to_num(sector_logits, 0.)
                logits_list.append(sector_logits)
            logits = torch.cat(logits_list)  # dim: 768

        else:
            raise NotImplementedError

        output_list.append(logits.detach().cpu().numpy())
    return output_list


def compute_logits(data_type, modality, *args):
    assert data_type in ['32', '64']
    assert modality in ['range', 'voxel', 'point_voxel']
    is_voxel = 'voxel' in modality
    dataset_name = TYPE2DATASET[data_type]
    dataset_config = DATASET_CONFIG[dataset_name]
    bs = MODAL2BATCHSIZE[modality]

    model = build_model(dataset_name, MODALITY2MODEL[modality], device='cuda')

    output = tuple()
    for data in args:
        all_logits_list = []
        for i in range(math.ceil(len(data) / bs)):
            batch = data[i * bs:(i + 1) * bs]
            if is_voxel:
                batch = [pcd2voxel(preprocess_pcd(pcd, **dataset_config)) for pcd in batch]
                batch = sparse_collate_fn(batch)
                batch = {k: v.cuda() if isinstance(v, (torch.Tensor, SparseTensor, PointTensor)) else v for k, v in
                         batch.items()}
                with torch.no_grad():
                    batch_out = model(batch, return_final_logits=True)
                    batch_out = batch2list(batch_out, AGG_TYPE, **dataset_config)
                    all_logits_list.extend(batch_out)
            else:
                batch = [preprocess_range(pcd, **dataset_config) for pcd in batch]
                batch = torch.from_numpy(np.stack(batch)).float().cuda()
                with torch.no_grad():
                    batch_out = model(batch, return_final_logits=True, agg_type=AGG_TYPE)
                    all_logits_list.append(batch_out)
        if is_voxel:
            all_logits = np.stack(all_logits_list)
        else:
            all_logits = np.vstack(all_logits_list)
        output += (all_logits,)

    del model, batch, batch_out
    torch.cuda.empty_cache()
    return output


def compute_pairwise_cd(x, y, module=None):
    if module is None:
        module = chamfer_3DDist()
    if x.ndim == 2 and y.ndim == 2:
        x, y = x[None], y[None]
    x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
    dist1, dist2, _, _ = module(x, y)
    dist = (dist1.mean() + dist2.mean()) / 2
    return dist.item()


def compute_pairwise_cd_batch(reference, samples):
    ndim = reference.ndim
    assert ndim in [2, 3]
    module = chamfer_3DDist() if ndim == 3 else chamfer_2DDist()
    len_r, len_s = reference.shape[0], [s.shape[0] for s in samples]
    max_len = max([len_r] + len_s)
    reference = torch.from_numpy(
        np.vstack([reference, np.ones((max_len - reference.shape[0], ndim), dtype=np.float32) * 1e6])).cuda()
    samples = [np.vstack([s, np.ones((max_len - s.shape[0], ndim), dtype=np.float32) * 1e6]) for s in samples]
    samples = torch.from_numpy(np.stack(samples)).cuda()
    reference = reference.expand_as(samples)
    dist_r, dist_s, _, _ = module(reference, samples)

    results = []
    for i in range(samples.shape[0]):
        dist1, dist2, len1, len2 = dist_r[i], dist_s[i], len_r, len_s[i]
        dist = (dist1[:len1].mean() + dist2[:len2].mean()) / 2.
        results.append(dist.item())
    return results


def compute_pairwise_emd(x, y, module=None):
    if module is None:
        module = emdModule()
    n_points = min(x.shape[0], y.shape[0])
    n_points = n_points - n_points % 1024
    x, y = x[:n_points], y[:n_points]
    if x.ndim == 2 and y.ndim == 2:
        x, y = x[None], y[None]
    x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
    dist, _ = module(x, y, 0.005, 50)
    dist = torch.sqrt(dist).mean()
    return dist.item()
