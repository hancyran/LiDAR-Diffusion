import math
import random
import warnings
from itertools import cycle
from typing import List, Optional, Tuple, Callable

from PIL import Image as pil_image, ImageDraw as pil_img_draw, ImageFont
from more_itertools.recipes import grouper
from .utils import COLOR_PALETTE, WHITE, GRAY_75, BLACK, additional_parameters_string, pad_list, get_circle_size, \
    get_plot_font_size, absolute_bbox
from ..helper_types import BoundingBox, Annotation, Image
from torch import LongTensor, Tensor
from torchvision.transforms import PILToTensor


pil_to_tensor = PILToTensor()


def convert_pil_to_tensor(image: Image) -> Tensor:
    with warnings.catch_warnings():
        # to filter PyTorch UserWarning as described here: https://github.com/pytorch/vision/issues/2194
        warnings.simplefilter("ignore")
        return pil_to_tensor(image)


class ObjectsCenterPointsConditionalBuilder:
    def __init__(self, no_object_classes: int, no_max_objects: int, no_tokens: int, num_beams: int):
        self.no_object_classes = no_object_classes
        self.no_max_objects = no_max_objects
        self.no_tokens = no_tokens
        # self.no_sections = int(math.sqrt(self.no_tokens))
        self.no_sections = (self.no_tokens // num_beams, num_beams)  # (width, height)

    @property
    def none(self) -> int:
        return self.no_tokens - 1

    @property
    def object_descriptor_length(self) -> int:
        return 2

    @property
    def empty_tuple(self) -> Tuple:
        return (self.none,) * self.object_descriptor_length

    @property
    def embedding_dim(self) -> int:
        return self.no_max_objects * self.object_descriptor_length

    def tokenize_coordinates(self, x: float, y: float) -> int:
        """
        Express 2d coordinates with one number.
        Example: assume self.no_tokens = 16, then no_sections = 4:
        0  0  0  0
        0  0  #  0
        0  0  0  0
        0  0  0  x
        Then the # position corresponds to token 6, the x position to token 15.
        @param x: float in [0, 1]
        @param y: float in [0, 1]
        @return: discrete tokenized coordinate
        """
        x_discrete = int(round(x * (self.no_sections[0] - 1)))
        y_discrete = int(round(y * (self.no_sections[1] - 1)))
        return y_discrete * self.no_sections[0] + x_discrete

    def coordinates_from_token(self, token: int) -> (float, float):
        x = token % self.no_sections[0]
        y = token // self.no_sections[0]
        return x / (self.no_sections[0] - 1), y / (self.no_sections[1] - 1)

    def bbox_from_token_pair(self, token1: int, token2: int) -> BoundingBox:
        x0, y0 = self.coordinates_from_token(token1)
        x1, y1 = self.coordinates_from_token(token2)
        # x2, y2 = self.coordinates_from_token(token3)
        # x3, y3 = self.coordinates_from_token(token4)
        return x0, y0, x1, y1

    def token_pair_from_bbox(self, bbox: BoundingBox) -> Tuple:
        # return self.tokenize_coordinates(bbox[0], bbox[1]), self.tokenize_coordinates(bbox[2], bbox[3]), self.tokenize_coordinates(bbox[4], bbox[5]), self.tokenize_coordinates(bbox[6], bbox[7])
        return self.tokenize_coordinates(bbox[0], bbox[1]), self.tokenize_coordinates(bbox[4], bbox[5])

    def inverse_build(self, conditional: LongTensor) \
            -> Tuple[List[Tuple[int, Tuple[float, float]]], Optional[BoundingBox]]:
        conditional_list = conditional.tolist()
        table_of_content = grouper(conditional_list, self.object_descriptor_length)
        assert conditional.shape[0] == self.embedding_dim
        return [
            (object_tuple[0], self.coordinates_from_token(object_tuple[1]))
            for object_tuple in table_of_content if object_tuple[0] != self.none
        ], None

    def plot(self, conditional: LongTensor, label_for_category_no: Callable[[int], str], figure_size: Tuple[int, int],
             line_width: int = 3, font_size: Optional[int] = None) -> Tensor:
        plot = pil_image.new('RGB', figure_size, WHITE)
        draw = pil_img_draw.Draw(plot)
        circle_size = get_circle_size(figure_size)
        # font = ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',
        #                           size=get_plot_font_size(font_size, figure_size))
        font = ImageFont.load_default()
        width, height = plot.size
        description, crop_coordinates = self.inverse_build(conditional)
        for (representation, (x, y)), color in zip(description, cycle(COLOR_PALETTE)):
            x_abs, y_abs = x * width, y * height
            ann = self.representation_to_annotation(representation)
            label = label_for_category_no(ann.category_id) + ' ' + additional_parameters_string(ann)
            ellipse_bbox = [x_abs - circle_size, y_abs - circle_size, x_abs + circle_size, y_abs + circle_size]
            draw.ellipse(ellipse_bbox, fill=color, width=0)
            draw.text((x_abs, y_abs), label, anchor='md', fill=BLACK, font=font)
        if crop_coordinates is not None:
            draw.rectangle(absolute_bbox(crop_coordinates, width, height), outline=GRAY_75, width=line_width)
        return convert_pil_to_tensor(plot) / 127.5 - 1.

    def object_representation(self, annotation: Annotation) -> int:
        return annotation.category_id

    def representation_to_annotation(self, representation: int) -> Annotation:
        category_id = representation % self.no_object_classes
        # noinspection PyTypeChecker
        return Annotation(
            bbox=None,
            category_id=category_id,
        )

    def _make_object_descriptors(self, annotations: List[Annotation]) -> List[Tuple[int, ...]]:
        object_tuples = [
            (self.object_representation(a),
             self.tokenize_coordinates(a.center[0], a.center[1]))
            for a in annotations
        ]
        empty_tuple = (self.none, self.none)
        object_tuples = pad_list(object_tuples, empty_tuple, self.no_max_objects)
        return object_tuples

    def build(self, annotations: List[Annotation]) \
            -> LongTensor:
        if len(annotations) == 0:
            warnings.warn('Did not receive any annotations.')

        random.shuffle(annotations)
        if len(annotations) > self.no_max_objects:
            warnings.warn('Received more annotations than allowed.')
            annotations = annotations[:self.no_max_objects]

        object_tuples = self._make_object_descriptors(annotations)
        flattened = [token for tuple_ in object_tuples for token in tuple_]
        assert len(flattened) == self.embedding_dim
        assert all(0 <= value < self.no_tokens for value in flattened)

        return LongTensor(flattened)
