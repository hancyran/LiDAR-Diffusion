import math
import sys

sys.path.append('./')

import os, argparse, glob, datetime, yaml
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import trange
import joblib

from omegaconf import OmegaConf
from PIL import Image

from lidm.models.diffusion.ddim import DDIMSampler
from lidm.utils.misc_utils import instantiate_from_config, set_seed, count_params
from lidm.utils.lidar_utils import range2pcd
from lidm.eval.eval_utils import evaluate

DATASET2METRICS = {'kitti': ['frid', 'fsvd', 'fpvd', 'jsd', 'mmd'], 'nuscenes': ['fsvd', 'fpvd', 'jsd', 'mmd']}
DATASET2TYPE = {'kitti': '64', 'nuscenes': '32'}


custom_to_range = lambda x: (x * 255.).clamp(0, 255).floor() / 255.


def custom_to_pcd(x, config):
    x = x.squeeze().detach().cpu().numpy()
    x = (np.clip(x, -1., 1.) + 1.) / 2.
    xyz, _, _ = range2pcd(x, **config['data']['params']['dataset'])

    rgb = np.zeros_like(xyz)
    return xyz, rgb


def custom_to_pil(x):
    x = x.detach().cpu().squeeze().numpy()
    # x = np.clip(x, 0., 1.)
    x = (np.clip(x, -1., 1.) + 1.) / 2.
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)

    return x


def custom_to_np(x):
    x = x.detach().cpu().squeeze().numpy()
    x = (np.clip(x, -1., 1.) + 1.) / 2.
    x = x.astype(np.float32)  # NOTE: use predicted continuous depth instead of np.uint8 depth
    return x


def logs2pil(logs, keys=["samples"]):
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


@torch.no_grad()
def convsample(model, shape, return_intermediates=True, verbose=True, make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape, return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(None, shape, verbose=verbose)


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0, verbose=False):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=verbose, disable_tqdm=True)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, image_size=None, vanilla=False, custom_steps=None, eta=1.0, verbose=False):
    log = dict()
    if image_size is None:
        image_size = model.model.diffusion_model.image_size
    shape = [batch_size, model.model.diffusion_model.in_channels, *image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape, make_prog_row=True, verbose=verbose)
        else:
            sample, intermediates = convsample_ddim(model, custom_steps, shape, eta, verbose)
        t1 = time.time()
    x_sample = model.decode_first_stage(sample)

    log["samples"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    if verbose:
        print(f'Throughput for this batch: {log["throughput"]}')
    return log


def run(model, imglogdir, pcdlogdir, batch_size=50, image_size=None, vanilla=False, custom_steps=None, eta=None, n_samples=50000,
        nplog=None, config=None, verbose=False):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(imglogdir, '*.png')))

    if model.cond_stage_model is None:
        all_samples = []
        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(math.ceil(n_samples / batch_size), desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size, image_size, vanilla, custom_steps, eta, verbose)
            n_saved = save_logs(logs, imglogdir, pcdlogdir, n_saved=n_saved, key="samples", config=config)
            all_samples.extend([custom_to_pcd(img, config)[0].astype(np.float32) for img in logs["samples"]])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        joblib.dump(all_samples, os.path.join(nplog, f"samples.pcd"))
    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"Sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")
    return all_samples


def save_logs(logs, imglogdir, pcdlogdir, n_saved=0, key="samples", np_path=None, config=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    # save as image
                    img = custom_to_pil(x)
                    imgpath = os.path.join(imglogdir, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    # save as point cloud
                    xyz, rgb = custom_to_pcd(x, config)
                    pcdpath = os.path.join(pcdlogdir, f"{key}_{n_saved:06}.txt")
                    np.savetxt(pcdpath, np.hstack([xyz, rgb]), fmt='%.3f')
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
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
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=1000
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
        '--image_size',
        nargs='+',
        type=int,
        default=None)
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
    count_params(model.first_stage_model, verbose=True)
    return model, global_step


def visualize(samples, logdir):
    pcdlogdir = os.path.join(logdir, "pcd")
    os.makedirs(pcdlogdir, exist_ok=True)
    for i, pcd in enumerate(samples):
        # save as point cloud
        pcdpath = os.path.join(pcdlogdir, f"{i:06}.txt")
        np.savetxt(pcdpath, pcd, fmt='%.3f')


def test_collate_fn(data):
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
        if opt.eval:
            assert len(opt.n_samples) == 2000, "Specify n_samples=2000 for evaluation"
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
        all_samples = run(model, imglogdir, pcdlogdir, eta=opt.eta, vanilla=opt.vanilla,
                          n_samples=opt.n_samples, custom_steps=opt.custom_steps,
                          batch_size=opt.batch_size, image_size=opt.image_size,
                          nplog=numpylogdir, config=config, verbose=opt.verbose)
        # recycle gpu memory
        del model
        torch.cuda.empty_cache()
    else:
        all_samples = joblib.load(opt.file)
        if opt.eval:
            assert len(all_samples) == 2000, "Prepare 2000 samples before evaluation"
        all_samples = [sample.astype(np.float32) for sample in all_samples]

    if opt.image_size is None:
        # traverse all validation data
        data_config = config['data']['params']['validation']
        data_config['params'].update({'dataset_config': config['data']['params']['dataset'],
                                      'aug_config': config['data']['params']['aug'], 'return_pcd': True})
        dataset = instantiate_from_config(data_config)
        dataloader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False, drop_last=False,
                                collate_fn=test_collate_fn)
        all_gt = []
        for batch in dataloader:
            all_gt.extend(batch)

    # evaluation
    if opt.eval:
        metrics, data_type = DATASET2METRICS[opt.dataset], DATASET2TYPE[opt.dataset]
        evaluate(all_gt, all_samples, metrics, data_type)
