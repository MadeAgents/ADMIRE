from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing import Any, Dict, List, Optional, Union
from verl.models.transformers.qwen2_vl import get_rope_index
import verl.utils.torch_functional as verl_F

import io
import math
import torch

from ..torch_functional import postprocess_data


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], Image.Image]) -> Image.Image:
        if isinstance(image, dict):
            image = Image.open(io.BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class MessagesDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        max_prompt_length: int = 1024,
        truncation: str = "error",
        fast_rollout: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.fast_rollout = fast_rollout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        messages = self.data[index]
        processor = self.processor

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        images, _ = process_vision_info(messages, return_video_kwargs=False)

        sample = dict()
        sample["multi_modal_data"] = {"image": images}  # [PIL.Image, ...]

        if not self.fast_rollout:
            # Multi-turn conversation tokenization
            inputs = processor(images=images, text=[prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = inputs.pop("input_ids")[0]
            attention_mask = inputs.pop("attention_mask")[0]
            sample["multi_modal_inputs"] = dict(inputs)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            # to make sure dataproto can be created
            input_ids = torch.zeros((0,), dtype=torch.int64)
            attention_mask = torch.zeros((0,), dtype=torch.int64)
            position_ids = torch.zeros((3, 0), dtype=torch.int64)
            sample["multi_modal_inputs"] = dict()

        input_ids, attention_mask, position_ids = postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["position_ids"] = position_ids
        sample["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        return sample


def collate_fn_dummy(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return features

class MessagesDataset2(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        max_prompt_length: int = 1024,
        truncation: str = "error",
        fast_rollout: bool = False,
        use_human_helps: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.fast_rollout = fast_rollout
        self.use_human_helps = use_human_helps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        messages = self.data[index]

        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        images, _ = process_vision_info(messages, return_video_kwargs=False)

        sample = dict()
        sample["multi_modal_data"] = {"image": images}  # [PIL.Image, ...]

        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if self.use_human_helps:
            if messages[-2]["role"] == "user" and messages[-1]["role"] == "user":
                raw_prompt_wo_human_helps = self.processor.apply_chat_template(messages[: -1], add_generation_prompt=False, tokenize=False)
                input_ids_wo_human_helps = self.processor(text=[raw_prompt_wo_human_helps], images=images, return_tensors="pt")["input_ids"]
                human_helps_length = len(input_ids[0]) - len(input_ids_wo_human_helps[0])  # with generation prompt
            else:
                human_helps_length = 0

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        sample["multi_modal_inputs"] = dict(model_inputs)

        # second_per_grid_ts isn't used for training, just for mrope
        sample["multi_modal_inputs"].pop("second_per_grid_ts", None)

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]  # (1, 3, seq_len)

        sample["input_ids"] = input_ids[0]
        sample["attention_mask"] = attention_mask[0]
        sample["position_ids"] = position_ids[0]

        if self.use_human_helps:
            sample["attention_mask_wo_human_helps"] = attention_mask[0].clone()
            if human_helps_length > 3:
                sample["attention_mask_wo_human_helps"][-human_helps_length: -3] *= 0  # the length of generation prompt is 3

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        sample["raw_prompt_ids"] = raw_prompt_ids
        return sample