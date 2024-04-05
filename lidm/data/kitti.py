import glob
import os
import pickle
import numpy as np
import yaml
from PIL import Image
import xml.etree.ElementTree as ET

from lidm.data.base import DatasetBase
from .annotated_dataset import Annotated3DObjectsDataset
from .conditional_builder.utils import corners_3d_to_2d
from .helper_types import Annotation
from ..utils.lidar_utils import pcd2range, pcd2coord2d, range2pcd

# TODO add annotation categories and semantic categories
CATEGORIES = ['ignore', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
              'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
              'pole', 'traffic-sign']
CATE2LABEL = {k: v for v, k in enumerate(CATEGORIES)}  # 0: invalid, 1~10: categories
LABEL2RGB = np.array([(0, 0, 0), (0, 0, 142), (119, 11, 32), (0, 0, 230), (0, 0, 70), (0, 0, 90), (220, 20, 60),
                      (255, 0, 0), (0, 0, 110), (128, 64, 128), (250, 170, 160), (244, 35, 232), (230, 150, 140),
                      (70, 70, 70), (190, 153, 153), (107, 142, 35), (0, 80, 100), (230, 150, 140), (153, 153, 153),
                      (220, 220, 0)])
CAMERAS = ['CAM_FRONT']
BBOX_CATS = ['car', 'people', 'cycle']
BBOX_CAT2LABEL = {'car': 0, 'truck': 0, 'bus': 0, 'caravan': 0, 'person': 1, 'rider': 2, 'motorcycle': 2, 'bicycle': 2}

# train + test
SEM_KITTI_TRAIN_SET = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
KITTI_TRAIN_SET = SEM_KITTI_TRAIN_SET + ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
KITTI360_TRAIN_SET = ['00', '02', '04', '05', '06', '07', '09', '10'] + ['08']  # partial test data at '02' sequence
CAM_KITTI360_TRAIN_SET = ['00', '04', '05', '06', '07', '08', '09', '10']  # cam mismatch lidar in '02'

# validation
SEM_KITTI_VAL_SET = KITTI_VAL_SET = ['08']
CAM_KITTI360_VAL_SET = KITTI360_VAL_SET = ['03']


class KITTIBase(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = 'kitti'
        self.num_sem_cats = kwargs['dataset_config'].num_sem_cats + 1

    @staticmethod
    def load_lidar_sweep(path):
        scan = np.fromfile(path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    def load_semantic_map(self, path, pcd):
        raise NotImplementedError

    def load_camera(self, path):
        raise NotImplementedError

    def __getitem__(self, idx):
        example = dict()
        data_path = self.data[idx]
        # lidar point cloud
        sweep = self.load_lidar_sweep(data_path)

        if self.lidar_transform:
            sweep, _ = self.lidar_transform(sweep, None)

        if self.condition_key == 'segmentation':
            # semantic maps
            proj_range, sem_map = self.load_semantic_map(data_path, sweep)
            example[self.condition_key] = sem_map
        else:
            proj_range, _ = pcd2range(sweep, self.img_size, self.fov, self.depth_range)
        proj_range, proj_mask = self.process_scan(proj_range)
        example['image'], example['mask'] = proj_range, proj_mask
        if self.return_pcd:
            reproj_sweep, _, _ = range2pcd(proj_range[0] * .5 + .5, self.fov, self.depth_range, self.depth_scale, self.log_scale)
            example['raw'] = sweep
            example['reproj'] = reproj_sweep.astype(np.float32)

        # image degradation
        if self.degradation_transform:
            degraded_proj_range = self.degradation_transform(proj_range)
            example['degraded_image'] = degraded_proj_range

        # cameras
        if self.condition_key == 'camera':
            cameras = self.load_camera(data_path)
            example[self.condition_key] = cameras

        return example


class SemanticKITTIBase(KITTIBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.condition_key in ['segmentation']  # for segmentation input only
        self.label2rgb = LABEL2RGB

    def prepare_data(self):
        # read data paths from KITTI
        for seq_id in eval('SEM_KITTI_%s_SET' % self.split.upper()):
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'dataset/sequences/{seq_id}/velodyne/*.bin')))
        # read label mapping
        data_config = yaml.safe_load(open('./data/config/semantic-kitti.yaml', 'r'))
        remap_dict = data_config["learning_map"]
        max_key = max(remap_dict.keys())
        self.learning_map = np.zeros((max_key + 100), dtype=np.int32)
        self.learning_map[list(remap_dict.keys())] = list(remap_dict.values())

    def load_semantic_map(self, path, pcd):
        label_path = path.replace('velodyne', 'labels').replace('.bin', '.label')
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF  # semantic label in lower half
        labels = self.learning_map[labels]

        proj_range, sem_map = pcd2range(pcd, self.img_size, self.fov, self.depth_range, labels=labels)
        # sem_map = np.expand_dims(sem_map, axis=0).astype(np.int64)
        sem_map = sem_map.astype(np.int64)
        if self.filtered_map_cats is not None:
            sem_map[np.isin(sem_map, self.filtered_map_cats)] = 0  # set filtered category as noise
        onehot = np.eye(self.num_sem_cats, dtype=np.float32)[sem_map].transpose(2, 0, 1)
        return proj_range, onehot


class SemanticKITTITrain(SemanticKITTIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/SemanticKITTI', split='train', **kwargs)


class SemanticKITTIValidation(SemanticKITTIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/SemanticKITTI', split='val', **kwargs)


class KITTI360Base(KITTIBase):
    def __init__(self, split_per_view=None, **kwargs):
        super().__init__(**kwargs)
        self.split_per_view = split_per_view
        if self.condition_key == 'camera':
            assert self.split_per_view is not None, 'For camera-to-lidar, need to specify split_per_view'

    def prepare_data(self):
        # read data paths
        self.data = []
        if self.condition_key == 'camera':
            seq_list = eval('CAM_KITTI360_%s_SET' % self.split.upper())
        else:
            seq_list = eval('KITTI360_%s_SET' % self.split.upper())
        for seq_id in seq_list:
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'data_3d_raw/2013_05_28_drive_00{seq_id}_sync/velodyne_points/data/*.bin')))

    def random_drop_camera(self, camera_list):
        if np.random.rand() < self.aug_config['camera_drop'] and self.split == 'train':
            camera_list = [np.zeros_like(c) if i != len(camera_list) // 2 else c for i, c in enumerate(camera_list)]  # keep the middle view only
        return camera_list

    def load_camera(self, path):
        camera_path = path.replace('data_3d_raw', 'data_2d_camera').replace('velodyne_points/data', 'image_00/data_rect').replace('.bin', '.png')
        camera = np.array(Image.open(camera_path)).astype(np.float32) / 255.
        camera = camera.transpose(2, 0, 1)
        if self.view_transform:
            camera = self.view_transform(camera)
        camera_list = np.split(camera, self.split_per_view, axis=2)  # split into n chunks as different views
        camera_list = self.random_drop_camera(camera_list)
        return camera_list


class KITTI360Train(KITTI360Base):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/KITTI-360', split='train', **kwargs)


class KITTI360Validation(KITTI360Base):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/KITTI-360', split='val', **kwargs)


class AnnotatedKITTI360Base(Annotated3DObjectsDataset, KITTI360Base):
    def __init__(self, **kwargs):
        self.id_bbox_dict = dict()
        self.id_label_dict = dict()

        Annotated3DObjectsDataset.__init__(self, **kwargs)
        KITTI360Base.__init__(self, **kwargs)
        assert self.condition_key in ['center', 'bbox']  # for annotated images only

    @staticmethod
    def parseOpencvMatrix(node):
        rows = int(node.find('rows').text)
        cols = int(node.find('cols').text)
        data = node.find('data').text.split(' ')

        mat = []
        for d in data:
            d = d.replace('\n', '')
            if len(d) < 1:
                continue
            mat.append(float(d))
        mat = np.reshape(mat, [rows, cols])
        return mat

    def parseVertices(self, child):
        transform = self.parseOpencvMatrix(child.find('transform'))
        R = transform[:3, :3]
        T = transform[:3, 3]
        vertices = self.parseOpencvMatrix(child.find('vertices'))
        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        return vertices

    def parse_bbox_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        bbox_dict = dict()
        label_dict = dict()
        for child in root:
            if child.find('transform') is None:
                continue

            label_name = child.find('label').text
            if label_name not in BBOX_CAT2LABEL:
                continue

            label = BBOX_CAT2LABEL[label_name]
            timestamp = int(child.find('timestamp').text)
            # verts = self.parseVertices(child)
            verts = self.parseOpencvMatrix(child.find('vertices'))[:8]
            if timestamp in bbox_dict:
                bbox_dict[timestamp].append(verts)
                label_dict[timestamp].append(label)
            else:
                bbox_dict[timestamp] = [verts]
                label_dict[timestamp] = [label]
        return bbox_dict, label_dict

    def prepare_data(self):
        KITTI360Base.prepare_data(self)

        self.data = [p for p in self.data if '2013_05_28_drive_0008_sync' not in p]  # remove unlabeled sequence 08
        seq_list = eval('KITTI360_%s_SET' % self.split.upper())
        for seq_id in seq_list:
            if seq_id != '08':
                xml_path = os.path.join(self.data_root, f'data_3d_bboxes/train/2013_05_28_drive_00{seq_id}_sync.xml')
                bbox_dict, label_dict = self.parse_bbox_xml(xml_path)
                self.id_bbox_dict[seq_id] = bbox_dict
                self.id_label_dict[seq_id] = label_dict

    def load_annotation(self, path):
        seq_id = path.split('/')[-4].split('_')[-2][-2:]
        timestamp = int(path.split('/')[-1].replace('.bin', ''))
        verts_list = self.id_bbox_dict[seq_id][timestamp]
        label_list = self.id_label_dict[seq_id][timestamp]

        if self.condition_key == 'bbox':
            points = np.stack(verts_list)
        elif self.condition_key == 'center':
            points = (verts_list[0] + verts_list[6]) / 2.
        else:
            raise NotImplementedError
        labels = np.array([label_list])
        if self.anno_transform:
            points, labels = self.anno_transform(points, labels)
        return points, labels

    def __getitem__(self, idx):
        example = dict()
        data_path = self.data[idx]

        # lidar point cloud
        sweep = self.load_lidar_sweep(data_path)

        # annotations
        bbox_points, bbox_labels = self.load_annotation(data_path)

        if self.lidar_transform:
            sweep, bbox_points = self.lidar_transform(sweep, bbox_points)

        # point cloud -> range
        proj_range, _ = pcd2range(sweep, self.img_size, self.fov, self.depth_range)
        proj_range, proj_mask = self.process_scan(proj_range)
        example['image'], example['mask'] = proj_range, proj_mask
        if self.return_pcd:
            example['reproj'] = sweep

        # annotation -> range
        # NOTE: do not need to transform bbox points along with lidar, since their coordinates are based on range-image space instead of 3D space
        proj_bbox_points, proj_bbox_labels = pcd2coord2d(bbox_points, self.fov, self.depth_range, labels=bbox_labels)
        builder = self.conditional_builders[self.condition_key]
        if self.condition_key == 'bbox':
            proj_bbox_points = corners_3d_to_2d(proj_bbox_points)
            annotations = [Annotation(bbox=bbox.flatten(), category_id=label) for bbox, label in
                           zip(proj_bbox_points, proj_bbox_labels)]
        else:
            annotations = [Annotation(center=center, category_id=label) for center, label in
                           zip(proj_bbox_points, proj_bbox_labels)]
        example[self.condition_key] = builder.build(annotations)

        return example


class AnnotatedKITTI360Train(AnnotatedKITTI360Base):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/KITTI-360', split='train', cats=BBOX_CATS, **kwargs)


class AnnotatedKITTI360Validation(AnnotatedKITTI360Base):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/KITTI-360', split='train', cats=BBOX_CATS, **kwargs)


class KITTIImageBase(KITTIBase):
    """
    Range ImageSet only combining KITTI-360 and SemanticKITTI

    #Samples (Training): 98014, #Samples (Val): 3511

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.condition_key in [None, 'image']  # for image input only

    def prepare_data(self):
        # read data paths from KITTI-360
        self.data = []
        for seq_id in eval('KITTI360_%s_SET' % self.split.upper()):
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'KITTI-360/data_3d_raw/2013_05_28_drive_00{seq_id}_sync/velodyne_points/data/*.bin')))

        # read data paths from KITTI
        for seq_id in eval('KITTI_%s_SET' % self.split.upper()):
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'SemanticKITTI/dataset/sequences/{seq_id}/velodyne/*.bin')))


class KITTIImageTrain(KITTIImageBase):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset', split='train', **kwargs)


class KITTIImageValidation(KITTIImageBase):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset', split='val', **kwargs)
