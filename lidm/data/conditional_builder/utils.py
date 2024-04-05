import importlib
from typing import List, Any, Tuple, Optional

import numpy as np
from ..helper_types import BoundingBox, Annotation

# source: seaborn, color palette tab10
COLOR_PALETTE = [(30, 118, 179), (255, 126, 13), (43, 159, 43), (213, 38, 39), (147, 102, 188),
                 (139, 85, 74), (226, 118, 193), (126, 126, 126), (187, 188, 33), (22, 189, 206)]
BLACK = (0, 0, 0)
GRAY_75 = (63, 63, 63)
GRAY_50 = (127, 127, 127)
GRAY_25 = (191, 191, 191)
WHITE = (255, 255, 255)
FULL_CROP = (0., 0., 1., 1.)


def corners_3d_to_2d(corners3d):
    """
    Args:
        corners3d: (N, 8, 2)
    Returns:
        corners2d: (N, 4, 2)
    """
    # select pairs to reorganize
    mask_0_3 = corners3d[:, 0:4, 0].argmax(1) // 2 != 0
    mask_4_7 = corners3d[:, 4:8, 0].argmin(1) // 2 != 0

    # reorganize corners in the order of (bottom-right, bottom-left)
    corners3d[mask_0_3, 0:4] = corners3d[mask_0_3][:, [2, 3, 0, 1]]
    # reorganize corners in the order of (top-left, top-right)
    corners3d[mask_4_7, 4:8] = corners3d[mask_4_7][:, [2, 3, 0, 1]]

    # calculate corners in order
    bot_r = np.stack([corners3d[:, 0:2, 0].max(1), corners3d[:, 0:2, 1].min(1)], axis=-1)
    bot_l = np.stack([corners3d[:, 2:4, 0].min(1), corners3d[:, 2:4, 1].min(1)], axis=-1)
    top_l = np.stack([corners3d[:, 4:6, 0].min(1), corners3d[:, 4:6, 1].max(1)], axis=-1)
    top_r = np.stack([corners3d[:, 6:8, 0].max(1), corners3d[:, 6:8, 1].max(1)], axis=-1)

    return np.stack([bot_r, bot_l, top_l, top_r], axis=1)


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (N, 3 + C)
        angle: angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones)).reshape((-1, 3, 3))
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        corners3d: (N, 8, 3)
    """
    template = np.array(
        [[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]],
    ) / 2

    # corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = np.tile(boxes3d[:, None, 3:6], (1, 8, 1)) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape((-1, 8, 3)), boxes3d[:, 6]).reshape((-1, 8, 3))
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def intersection_area(rectangle1: BoundingBox, rectangle2: BoundingBox) -> float:
    """
    Give intersection area of two rectangles.
    @param rectangle1: (x0, y0, w, h) of first rectangle
    @param rectangle2: (x0, y0, w, h) of second rectangle
    """
    rectangle1 = rectangle1[0], rectangle1[1], rectangle1[0] + rectangle1[2], rectangle1[1] + rectangle1[3]
    rectangle2 = rectangle2[0], rectangle2[1], rectangle2[0] + rectangle2[2], rectangle2[1] + rectangle2[3]
    x_overlap = max(0., min(rectangle1[2], rectangle2[2]) - max(rectangle1[0], rectangle2[0]))
    y_overlap = max(0., min(rectangle1[3], rectangle2[3]) - max(rectangle1[1], rectangle2[1]))
    return x_overlap * y_overlap


def horizontally_flip_bbox(bbox: BoundingBox) -> BoundingBox:
    return 1 - (bbox[0] + bbox[2]), bbox[1], bbox[2], bbox[3]


def absolute_bbox(relative_bbox: BoundingBox, width: int, height: int) -> Tuple[int, int, int, int]:
    bbox = relative_bbox
    # bbox = bbox[0] * width, bbox[1] * height, (bbox[0] + bbox[2]) * width, (bbox[1] + bbox[3]) * height
    bbox = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
    # return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x1, x2 = min(int(bbox[2]), int(bbox[0])), max(int(bbox[2]), int(bbox[0]))
    y1, y2 = min(int(bbox[3]), int(bbox[1])), max(int(bbox[3]), int(bbox[1]))
    if x1 == x2:
        x2 += 1
    if y1 == y2:
        y2 += 1
    return x1, y1, x2, y2


def pad_list(list_: List, pad_element: Any, pad_to_length: int) -> List:
    return list_ + [pad_element for _ in range(pad_to_length - len(list_))]


def rescale_annotations(annotations: List[Annotation], crop_coordinates: BoundingBox, flip: bool) -> \
        List[Annotation]:
    def clamp(x: float):
        return max(min(x, 1.), 0.)

    def rescale_bbox(bbox: BoundingBox) -> BoundingBox:
        x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
        y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
        w = min(bbox[2] / crop_coordinates[2], 1 - x0)
        h = min(bbox[3] / crop_coordinates[3], 1 - y0)
        if flip:
            x0 = 1 - (x0 + w)
        return x0, y0, w, h

    return [a._replace(bbox=rescale_bbox(a.bbox)) for a in annotations]


def filter_annotations(annotations: List[Annotation], crop_coordinates: BoundingBox) -> List:
    return [a for a in annotations if intersection_area(a.bbox, crop_coordinates) > 0.0]


def additional_parameters_string(annotation: Annotation, short: bool = True) -> str:
    sl = slice(1) if short else slice(None)
    string = ''
    if not (annotation.is_group_of or annotation.is_occluded or annotation.is_depiction or annotation.is_inside):
        return string
    if annotation.is_group_of:
        string += 'group'[sl] + ','
    if annotation.is_occluded:
        string += 'occluded'[sl] + ','
    if annotation.is_depiction:
        string += 'depiction'[sl] + ','
    if annotation.is_inside:
        string += 'inside'[sl]
    return '(' + string.strip(",") + ')'


def get_plot_font_size(font_size: Optional[int], figure_size: Tuple[int, int]) -> int:
    if font_size is None:
        font_size = 10
        if max(figure_size) >= 256:
            font_size = 12
        if max(figure_size) >= 512:
            font_size = 15
    return font_size


def get_circle_size(figure_size: Tuple[int, int]) -> int:
    circle_size = 2
    if max(figure_size) >= 256:
        circle_size = 3
    if max(figure_size) >= 512:
        circle_size = 4
    return circle_size


def load_object_from_string(object_string: str) -> Any:
    """
    Source: https://stackoverflow.com/a/10773699
    """
    module_name, class_name = object_string.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)
