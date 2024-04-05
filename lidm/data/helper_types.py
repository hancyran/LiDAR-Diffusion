from typing import Tuple, Optional, NamedTuple, Union, List
from PIL.Image import Image as pil_image
from torch import Tensor

try:
  from typing import Literal
except ImportError:
  from typing_extensions import Literal

Image = Union[Tensor, pil_image]
# BoundingBox = Tuple[float, float, float, float]  # x0, y0, w, h | x0, y0, x1, y1
# BoundingBox3D = Tuple[float, float, float, float, float, float]  # x0, y0, z0, l, w, h
BoundingBox = Tuple[float, float, float, float]  # corner coordinates (x,y) in the order of bottom-right -> bottom-left -> top-left -> top-right
Center = Tuple[float, float]


class Annotation(NamedTuple):
    category_id: int
    bbox: Optional[BoundingBox] = None
    center: Optional[Center] = None
