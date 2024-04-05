from pathlib import Path
from typing import Optional, List, Dict, Union, Any
import warnings

from torch.utils.data import Dataset

from .conditional_builder.objects_bbox import ObjectsBoundingBoxConditionalBuilder
from .conditional_builder.objects_center_points import ObjectsCenterPointsConditionalBuilder


class Annotated3DObjectsDataset(Dataset):
    def __init__(self, min_objects_per_image: int,
                 max_objects_per_image: int, no_tokens: int, num_beams: int, cats: List[str],
                 cat_blacklist: Optional[List[str]] = None, **kwargs):
        self.min_objects_per_image = min_objects_per_image
        self.max_objects_per_image = max_objects_per_image
        self.no_tokens = no_tokens
        self.num_beams = num_beams

        self.categories = [c for c in cats if c not in cat_blacklist] if cat_blacklist is not None else cats
        self._conditional_builders = None

    @property
    def no_classes(self) -> int:
        return len(self.categories)

    @property
    def conditional_builders(self) -> ObjectsCenterPointsConditionalBuilder:
        # cannot set this up in init because no_classes is only known after loading data in init of superclass
        if self._conditional_builders is None:
            self._conditional_builders = {
                'center': ObjectsCenterPointsConditionalBuilder(
                    self.no_classes,
                    self.max_objects_per_image,
                    self.no_tokens,
                    self.num_beams
                ),
                'bbox': ObjectsBoundingBoxConditionalBuilder(
                    self.no_classes,
                    self.max_objects_per_image,
                    self.no_tokens,
                    self.num_beams
                )
            }
        return self._conditional_builders

    def get_textual_label_for_category_id(self, category_id: int) -> str:
        return self.categories[category_id]
