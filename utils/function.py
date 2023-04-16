from enum import Enum
from typing import Any, List, Optional, Tuple, Union
import math
from torchvision.transforms import functional as F
import PIL
from PIL import Image, ImageEnhance, ImageOps


def get_dimensions(img):
    """
    Args:
        img: PIL Image

    Returns:

    """
    if hasattr(img, "getbands"):
        channels = len(img.getbands())
    else:
        channels = img.channels
    # channels = img.channels
    width, height = img.size
    return [channels, height, width]


def _get_inverse_affine_matrix(center: List[float], angle: float, translate: List[float], scale: float, shear: List[float]):
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix:  1, 0, tx
    #                                 0, 1, ty
    #                                 0, 0, 1
    #       C is translation matrix to keep center: [1, 0, cx
    #                                                0, 1, cy
    #                                                0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]
    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def affine(img, angle, translate, scale, shear, interpolation=Image.Resampling.NEAREST, fill=None, center=None):
    _, height, width = get_dimensions(img)
    # center = (width * 0.5 + 0.5, height * 0.5 + 0.5)
    # it is visually better to estimate the center without 0.5 offset
    # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
    if center is None:
        center = [width * 0.5, height * 0.5]
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    return img.transform(size=img.size, method=Image.AFFINE, data=matrix, resample=interpolation, fillcolor=fill)


def rotate(img: PIL.Image, angle: float, interpolation=Image.Resampling.NEAREST, expand=False, center=None, fill=None):
    return img.rotate(angle=angle, resample=interpolation, expand=expand, center=center, fillcolor=fill)


def adjust_brightness(img: PIL.Image, brightness_factor: float):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_saturation(img: PIL.Image, saturation_factor: float):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_contrast(img, contrast_factor: float):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_sharpness(img, sharpness_factor: float):
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness_factor)
    return img


def posterize(img, bits: int):
    """Posterize an image by reducing the number of bits for each color channel.
    """
    if not (0 <= bits <= 8):
        raise ValueError(f"The number if bits should be between 0 and 8. Got {bits}")

    return ImageOps.posterize(img, bits)


def solarize(img, threshold: float):
    """Solarize an RGB/grayscale image by inverting all pixel values above a threshold.
    """
    return ImageOps.solarize(img, int(threshold))


def autocontrast(img):
    """Maximize contrast of an image by remapping its
    pixels per channel so that the lowest becomes black and the lightest
    becomes white."""
    return ImageOps.autocontrast(img)


def equalize(img):
    """Equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output."""
    return ImageOps.equalize(img)


def invert(img):
    """Invert the colors of an RGB/grayscale image.

    Args:
        img (PIL Image or Tensor): Image to have its colors inverted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Returns:
        PIL Image or Tensor: Color inverted image.
    """
    return ImageOps.invert(img)
