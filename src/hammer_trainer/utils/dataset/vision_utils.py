from PIL import Image, ImageDraw
from typing import Union, List

import base64
import io
import numpy as np

from android_world.env.representation_utils import UIElement

def image_to_base64(image: Union[np.ndarray, Image.Image]):
    assert isinstance(image, (np.ndarray, Image.Image))
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    image_base64 = base64.b64encode(buffered.getvalue())
    # in py3, b64encode() returns bytes
    return f"""data:image/png;base64,{image_base64.decode("utf-8")}"""


def base64_to_image(image_base64):
    # Remove data URI prefix if present
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",")[1]
    # Decode the Base64 string
    image_base64 = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_base64))
    return image

def draw_interactive_ui_elements_on_screenshot(screenshot: Image, ui_elements: List[UIElement]):
    draw = ImageDraw.Draw(screenshot)
    for e in ui_elements:
        # if e.is_enabled and (e.is_checkable or e.is_clickable or e.is_editable or e.is_focusable): ## is_xxable info. sometimes missing
        if e.is_enabled:
            try:
                bbox = e.bbox_pixels
                draw.rectangle((bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max), outline='red', width=3)
            except:
                continue
    return screenshot