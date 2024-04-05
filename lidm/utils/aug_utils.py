import numpy as np


def get_lidar_transform(config, split):
    transform_list = []
    if config['rotate']:
        transform_list.append(RandomRotateAligned())
    if config['flip']:
        transform_list.append(RandomFlip())
    return Compose(transform_list) if len(transform_list) > 0 and split == 'train' else None


def get_camera_transform(config, split):
    # import open_clip
    # transform = open_clip.image_transform((224, 224), split == 'train', resize_longest_max=True)
    # TODO
    transform = None
    return transform


def get_anno_transform(config, split):
    if config['keypoint_drop'] and split == 'train':
        drop_range = config['keypoint_drop_range'] if 'keypoint_drop_range' in config else (5, 60)
        transform = RandomKeypointDrop(drop_range)
    else:
        transform = None
    return transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pcd, pcd1=None):
        for t in self.transforms:
            pcd, pcd1 = t(pcd, pcd1)
        return pcd, pcd1


class RandomFlip(object):
    def __init__(self, p=1.):
        self.p = p

    def __call__(self, coord, coord1=None):
        if np.random.rand() < self.p:
            if np.random.rand() < 0.5:
                coord[:, 0] = -coord[:, 0]
                if coord1 is not None:
                    coord1[:, 0] = -coord1[:, 0]
            if np.random.rand() < 0.5:
                coord[:, 1] = -coord[:, 1]
                if coord1 is not None:
                    coord1[:, 1] = -coord1[:, 1]
        return coord, coord1


class RandomRotateAligned(object):
    def __init__(self, rot=np.pi / 4, p=1.):
        self.rot = rot
        self.p = p

    def __call__(self, coord, coord1=None):
        if np.random.rand() < self.p:
            angle_z = np.random.uniform(-self.rot, self.rot)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            coord = np.dot(coord, R)
            if coord1 is not None:
                coord1 = np.dot(coord1, R)
        return coord, coord1


class RandomKeypointDrop(object):
    def __init__(self, num_range=(5, 60), p=.5):
        self.num_range = num_range
        self.p = p

    def __call__(self, center, category=None):
        if np.random.rand() < self.p:
            num = len(center)
            if num > self.num_range[0]:
                num_kept = np.random.randint(self.num_range[0], min(self.num_range[1], num))
                idx_kept = np.random.choice(num, num_kept, replace=False)
                center, category = center[idx_kept], category[idx_kept]
        return center, category


# class ResizeMaxSize(object):
#     def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
#         super().__init__()
#         if not isinstance(max_size, int):
#             raise TypeError(f"Size should be int. Got {type(max_size)}")
#         self.max_size = max_size
#         self.interpolation = interpolation
#         self.fn = min if fn == 'min' else min
#         self.fill = fill
#
#     def forward(self, img):
#         width, height = img.size
#         scale = self.max_size / float(max(height, width))
#         if scale != 1.0:
#             new_size = tuple(round(dim * scale) for dim in (height, width))
#             img = F.resize(img, new_size, self.interpolation)
#             pad_h = self.max_size - new_size[0]
#             pad_w = self.max_size - new_size[1]
#             img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
#         return img
