import math
import sys

sys.path.append('./')

import os, argparse, glob, datetime, yaml
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import joblib

from omegaconf import OmegaConf
from PIL import Image

from lidm.utils.misc_utils import instantiate_from_config, set_seed
from lidm.utils.lidar_utils import range2pcd
from lidm.eval.eval_utils import evaluate

# remove annoying user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATASET2METRICS = {'kitti': ['frid', 'fsvd', 'fpvd', 'jsd', 'mmd'], 'nuscenes': ['fsvd', 'fpvd']}
DATASET2TYPE = {'kitti': '64', 'nuscenes': '32'}

custom_to_range = lambda x: (x * 255.).clamp(0, 255).floor() / 255.


def custom_to_pcd(x, config, rgb=None):
    x = x.squeeze().detach().cpu().numpy()
    x = (np.clip(x, -1., 1.) + 1.) / 2.
    if rgb is not None:
        rgb = rgb.squeeze().detach().cpu().numpy()
        rgb = (np.clip(rgb, -1., 1.) + 1.) / 2.
        rgb = rgb.transpose(1, 2, 0)
    xyz, rgb, _ = range2pcd(x, color=rgb, **config['data']['params']['dataset'])

    return xyz, rgb


def custom_to_pil(x):
    x = x.detach().cpu().squeeze().numpy()
    x = (np.clip(x, -1., 1.) + 1.) / 2.
    x = (255 * x).astype(np.uint8)

    if x.ndim == 3:
        x = x.transpose(1, 2, 0)
    x = Image.fromarray(x)

    return x


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


def run(model, dataloader, imglogdir, pcdlogdir, nplog=None, config=None, verbose=False, log_config={}):
    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(imglogdir, '*.png')))

    all_samples, all_gt = [], []
    print(f"Running conditional sampling")
    for batch in tqdm(dataloader, desc="Sampling Batches (conditional)"):
        all_gt.extend(batch['reproj'])
        N = len(batch['reproj'])
        logs = model.log_images(batch, N=N, split='val', **log_config)
        n_saved = save_logs(logs, imglogdir, pcdlogdir, N, n_saved=n_saved, config=config)
        all_samples.extend([custom_to_pcd(img, config)[0].astype(np.float32) for img in logs["samples"]])
    joblib.dump(all_samples, os.path.join(nplog, f"samples.pcd"))

    print(f"Sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")
    return all_samples, all_gt


def save_logs(logs, imglogdir, pcdlogdir, num, n_saved=0, key_list=None, config=None):
    key_list = logs.keys() if key_list is None else key_list
    for i in range(num):
        for k in key_list:
            if k in ['reconstruction']:
                continue
            x = logs[k][i]
            # save as image
            if x.ndim == 3 and x.shape[0] in [1, 3]:
                img = custom_to_pil(x)
                imgpath = os.path.join(imglogdir, f"{k}_{n_saved:06}.png")
                img.save(imgpath)
            # save as point cloud
            if k in ['samples', 'inputs']:
                if config.model.params.cond_stage_key == 'segmentation':
                    xyz, rgb = custom_to_pcd(x, config, logs['original_conditioning'][i])
                else:
                    xyz, rgb = custom_to_pcd(x, config)
                pcdpath = os.path.join(pcdlogdir, f"{k}_{n_saved:06}.txt")
                np.savetxt(pcdpath, np.hstack([xyz, rgb]), fmt='%.3f')
        n_saved += 1
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
        default="none"
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "--vanilla",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "-f",
        "--file",
        help="the file path of samples",
        default=None
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="the numpy file path",
        default=1000
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset name [nuscenes, kitti]",
        required=True
    )
    parser.add_argument(
        "--baseline",
        default=False,
        action='store_true',
        help="baseline provided by other sources (default option is not baseline)?",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action='store_true',
        help="print status?",
    )
    parser.add_argument(
        "--eval",
        default=False,
        action='store_true',
        help="evaluation results?",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])
    return model, global_step


def visualize(samples, logdir):
    pcdlogdir = os.path.join(logdir, "pcd")
    os.makedirs(pcdlogdir, exist_ok=True)
    for i, pcd in enumerate(samples):
        # save as point cloud
        pcdpath = os.path.join(pcdlogdir, f"{i:06}.txt")
        np.savetxt(pcdpath, pcd, fmt='%.3f')


def test_collate_fn(data):
    output = {}
    keys = data[0].keys()
    for k in keys:
        v = [d[k] for d in data]
        if k not in ['reproj', 'raw']:
            v = torch.from_numpy(np.stack(v, 0))
        else:
            v = [d[k] for d in data]
        output[k] = v
    return output


def traverse_collate_fn(data):
    pcd_list = [example['reproj'].astype(np.float32) for example in data]
    return pcd_list


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None
    set_seed(opt.seed)

    if not os.path.exists(opt.resume) and not os.path.exists(opt.file):
        raise FileNotFoundError
    if os.path.isfile(opt.resume):
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    elif os.path.isfile(opt.file):
        try:
            logdir = '/'.join(opt.file.split('/')[:-5])
            if len(logdir) == 0:
                logdir = '/'.join(opt.file.split('/')[:-1])
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -5  # take a guess: path/to/logdir/samples/step_num/date/numpy/*.npz
            logdir = "/".join(paths[:idx])
        ckpt = None
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    if not opt.baseline:
        base_configs = [f'{logdir}/config.yaml']
    else:
        base_configs = [f'models/baseline/{opt.dataset}/template/config.yaml']
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True
    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    if opt.file is None:
        model, global_step = load_model(config, ckpt)
        print(f"global step: {global_step}")
        print(75 * "=")
        print("logging to:")
        logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
        imglogdir = os.path.join(logdir, "img")
        pcdlogdir = os.path.join(logdir, "pcd")
        numpylogdir = os.path.join(logdir, "numpy")

        os.makedirs(imglogdir)
        os.makedirs(pcdlogdir)
        os.makedirs(numpylogdir)
        print(logdir)
        print(75 * "=")

        # write config out
        sampling_file = os.path.join(logdir, "sampling_config.yaml")
        sampling_conf = vars(opt)

        with open(sampling_file, 'w') as f:
            yaml.dump(sampling_conf, f, default_flow_style=False)
        print(sampling_conf)

        # traverse all validation data
        data_config = config['data']['params']['validation']
        data_config['params'].update({'dataset_config': config['data']['params']['dataset'],
                                      'aug_config': config['data']['params']['aug'], 'return_pcd': True,
                                      'max_objects_per_image': 5})
        dataset = instantiate_from_config(data_config)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=False, drop_last=False,
                                collate_fn=test_collate_fn)

        # settings
        log_config = {'sample': True, 'ddim_steps': opt.custom_steps,
                      'quantize_denoised': False, 'inpaint': False, 'plot_progressive_rows': False,
                      'plot_diffusion_rows': False, 'dset': dataset}
        # test = dataset[0]
        all_samples, all_gt = run(model, dataloader, imglogdir, pcdlogdir, nplog=numpylogdir,
                                  config=config, verbose=opt.verbose, log_config=log_config)

        # recycle gpu memory
        del model
        torch.cuda.empty_cache()
    else:
        all_samples = joblib.load(opt.file)
        all_samples = [sample.astype(np.float32) for sample in all_samples]

        # traverse all validation data
        data_config = config['data']['params']['validation']
        data_config['params'].update({'dataset_config': config['data']['params']['dataset'],
                                      'aug_config': config['data']['params']['aug'], 'return_pcd': True})
        dataset = instantiate_from_config(data_config)
        dataloader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False, drop_last=False,
                                collate_fn=traverse_collate_fn)
        all_gt = []
        for batch in dataloader:
            all_gt.extend(batch)

    # evaluation
    if opt.eval:
        metrics, data_type = DATASET2METRICS[opt.dataset], DATASET2TYPE[opt.dataset]
        evaluate(all_gt, all_samples, metrics, data_type)
