#!/usr/bin/env python3
from pickle import FALSE

import numpy as np
import matplotlib
import pandas as pd
import os
import yaml
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple
import PIL
from PIL import ImageDraw, ImageFont

def add_axes_and_scale_bar(
    image: PIL.Image.Image,
    scale_bar_length: int,
    bin_resolution: float,
    axes_origin: tuple = (20, 20),
    scale_bar_pos: tuple = (10, 70),
    font: ImageFont.ImageFont = None
) -> PIL.Image.Image:
    """
    Draws x/y axes and a scale bar on the given PIL image.

    Args:
        image: PIL Image to draw on.
        scale_bar_length: Length of the scale bar in pixels.
        bin_resolution: Size of one pixel in nm.
        axes_origin: (x, y) position for the axes origin.
        scale_bar_pos: (x, y) position for the scale bar.
        font: Optional PIL font.

    Returns:
        Modified PIL Image.
    """
    imp = image.copy()
    draw = ImageDraw.Draw(imp)
    if font is None:
        font = ImageFont.load_default()

    # Axes
    ox, oy = axes_origin
    draw.text((ox - 10, oy + 20), "X", fill="white", font=font)
    draw.text((ox + 20, oy - 10), "Y", fill="white", font=font)
    draw.line((ox, oy, ox + 30, oy), fill="white", width=1)  # X-axis
    draw.line((ox, oy, ox, oy + 30), fill="white", width=1)  # Y-axis
    draw.polygon([(ox + 30, oy), (ox + 27, oy - 3), (ox + 27, oy + 3)], fill="white")  # X arrow
    draw.polygon([(ox, oy + 30), (ox - 3, oy + 27), (ox + 3, oy + 27)], fill="white")  # Y arrow

    # Scale bar
    sx, sy = scale_bar_pos
    scale_bar_height = 3
    draw.rectangle([sx, sy, sx + scale_bar_length, sy + scale_bar_height], fill="white")
    draw.text((sx, sy + scale_bar_height + 5), f"{scale_bar_length * bin_resolution:.0f} nm", fill="white", font=font)

    return imp